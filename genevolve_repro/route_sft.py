"""Four-rank Qwen3-VL-2B Route-SFT on the reviewed v3 examples."""

from contextlib import nullcontext
from functools import partial
import random
from statistics import fmean

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .common import model_ready, read_json, write_json
from .distributed import (
    barrier, export_policy, gather, initialize, latest, load_checkpoint, load_model,
    make_sharded_actor, restore_optimizer, restore_rng, save_checkpoint,
)
from .policy import ActorScorer
from .route_data import (
    MARKER_ID, load_pack, pack_manifest, parent_tokenizer, recipe, register_marker,
    route_identity, source_checkpoint_lock, verify_artifact_identity, verify_parent,
    verify_parent_identity, verify_tokenizer,
)
from .sft_export import sft_manifest
from .route_type import TYPE_TOKENS, action_boundary
from .route_objective import split_ce, supervision_layout
from .marker_seed import apply_marker_seed, prepare_marker_seed


def route_bce(logits, marker_id, is_bad):
    logits = logits.float()
    other = torch.logsumexp(torch.cat((logits[:marker_id], logits[marker_id + 1:])), dim=0)
    binary = logits[marker_id] - other
    return F.binary_cross_entropy_with_logits(binary, binary.new_tensor(float(is_bad))), binary.sigmoid()


class RouteScorer(ActorScorer):
    def __init__(self, model, projection_chunk=128, type_tokens=TYPE_TOKENS, initial_marker=None):
        super().__init__(model, projection_chunk)
        self.type_tokens = type_tokens
        if initial_marker is not None:
            apply_marker_seed(model, initial_marker, MARKER_ID)

    def forward(self, input_ids, start, is_bad, marker_id, route_lambda,
                *, tag_end=None, kind_weight=1.0, content_weight=0.15,
                tag_weight=0.5, with_metrics=False):
        hidden = self.model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        # Only the supervised token predictors are projected. H/R remain in
        # the attention context; no CE is imposed on their token positions.
        states = hidden.last_hidden_state[0, start - 1:-1]
        targets = input_ids[0, start:]

        def ce_tokens(state, ids):
            return F.cross_entropy(self.model.lm_head(state).float(), ids, reduction="none")

        parts = [checkpoint(ce_tokens, chunk, ids, use_reentrant=False) if torch.is_grad_enabled()
                 else ce_tokens(chunk, ids)
                 for chunk, ids in zip(states.split(self.projection_chunk), targets.split(self.projection_chunk))]
        token_ce = torch.cat(parts)
        ce = token_ce.mean()  # Original suffix mean, retained as a diagnostic only.
        bce, probability = route_bce(self.model.lm_head(states[0]), marker_id, is_bad)
        if tag_end is None:
            raise ValueError("Pass the tokenizer-verified action_tag_end; do not guess a tag span.")
        action_start, _ = action_boundary(input_ids[0], start, is_bad, marker_id, self.type_tokens)
        weighted_ce, metrics = split_ce(token_ce, action_start - start, tag_end - start,
                                        kind_weight, content_weight, tag_weight)
        loss = weighted_ce + route_lambda * bce
        result = (loss, ce.detach(), bce.detach(), probability.detach())
        return (*result, metrics) if with_metrics else result


def update(actor, optimizer, rows, batch, cfg, rank, size, device):
    batch_local = batch[rank::size]
    if len(batch_local) != 2 or cfg["micro_batch"] != 1:
        raise ValueError("Expected two unpadded samples on each of four ranks.")
    optimizer.zero_grad(set_to_none=True)
    stats = []
    for i, k in enumerate(batch_local):
        row = rows[k]
        ctx = actor.no_sync() if hasattr(actor, "no_sync") and i + 1 < len(batch_local) else nullcontext()
        with ctx:
            ids = torch.tensor([row["input_ids"]], device=device, dtype=torch.long)
            loss, ce, bce, p, ce_stats = actor(
                ids, row["target_start"], row["route_label"], MARKER_ID, cfg["route_lambda"],
                tag_end=row["action_tag_end"], kind_weight=row["action_kind_weight"],
                content_weight=cfg["content_ce_weight"], tag_weight=cfg["action_tag_ce_weight"],
                with_metrics=True,
            )
            finite = torch.isfinite(loss.detach()).int()
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite Route-SFT loss; the last saved epoch is unchanged.")
            # FSDP averages across ranks, each rank contributes two rows.
            (loss / len(batch_local)).backward()
        _, is_answer = action_boundary(row["input_ids"], row["target_start"],
                                       row["route_label"], MARKER_ID)
        stats.append({"route_label": row["route_label"], "is_answer": is_answer,
                      "loss": loss.detach().item(), "suffix_ce": ce.item(),
                      "route_bce": bce.item(), "p_rethink": p.item(),
                      **{key: value.item() for key, value in ce_stats.items()}})
    clip = getattr(actor, "clip_grad_norm_", None)
    norm = clip(cfg["clip_grad"]) if clip else torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg["clip_grad"])
    if not torch.isfinite(norm).item():
        raise FloatingPointError("Non-finite Route-SFT gradient; the last saved epoch is unchanged.")
    optimizer.step()
    return stats, float(norm)


def summarize_losses(flat, cfg):
    """One global sample mean, after rank gathering; never divide by B twice."""
    metrics = {key: fmean(x[key] for x in flat) for key in (
        "loss", "suffix_ce", "route_bce", "content_ce", "action_tag_ce",
        "content_weighted", "action_tag_weighted", "weighted_ce",
    )}
    metrics["route_weighted"] = cfg["route_lambda"] * metrics["route_bce"]
    for label, name in ((0, "good"), (1, "bad")):
        values = [x["p_rethink"] for x in flat if x["route_label"] == label]
        metrics[f"{name}_p_rethink"] = fmean(values) if values else None
    for is_answer, name in ((False, "tool"), (True, "answer")):
        items = [x for x in flat if x["is_answer"] == is_answer]
        metrics[f"target_{name}_count"] = len(items)
        metrics[f"target_{name}_tag_ce"] = fmean(x["action_tag_ce"] for x in items) if items else None
    return metrics


def run_epoch_export(out, step, actor, parent, rank, contract):
    from transformers import AutoProcessor

    folder = out / f"hf-step-{step:05d}"
    if rank == 0:
        already = model_ready(folder) and (folder / "completed.json").is_file()
    else:
        already = None
    if gather(already)[0]:
        if rank == 0:
            done = read_json(folder / "completed.json")
            if done != {"step": step, "algorithm": "route-sft", "contract": contract}:
                raise ValueError("Existing Route-SFT export belongs to an incompatible run.")
            receipt = folder / "export_sha256.json"
            if not receipt.is_file():
                write_json(receipt, sft_manifest(folder))
        barrier()
        return
    replica = load_model(parent, torch.bfloat16, attention="sdpa") if rank == 0 else None
    processor = AutoProcessor.from_pretrained(parent) if rank == 0 else None
    if rank == 0:
        register_marker(processor.tokenizer)
    export_policy(folder, actor, replica, processor, step, rank,
                  metadata={"algorithm": "route-sft", "contract": contract})
    if rank == 0:
        write_json(folder / "export_sha256.json", sft_manifest(folder))
    barrier()


def verify_route_export(path, cfg=None, expected_contract=None):
    cfg = cfg or recipe()
    total = cfg["batches_per_epoch"] * cfg["epochs"]
    done = read_json(path / "completed.json") if (path / "completed.json").is_file() else {}
    identity = done.get("contract") or {}
    if (done.get("algorithm") != "route-sft" or done.get("step") != total
            or identity.get("config") != cfg
            or identity.get("route_data_source_sha256") != source_checkpoint_lock()):
        raise ValueError("Stage 2 must start from the complete Route-SFT HF export for the configured epoch count.")
    verify_artifact_identity(identity.get("manifest_sha256"), identity.get("artifact_sha256"), cfg)
    verify_parent_identity(identity.get("parent"), cfg)
    if expected_contract is not None and identity != expected_contract:
        raise ValueError("Route-SFT export does not match the supplied 2B parent and v3 pack; use a new WORK.")
    receipt = read_json(path / "export_sha256.json") if (path / "export_sha256.json").is_file() else {}
    manifest = sft_manifest(path)
    if not receipt or manifest != receipt:
        raise ValueError("Final Route-SFT export SHA-256 does not match its receipt.")
    return {"completed": done, **manifest}


def _loop(work, parent, data, rank, size, device):
    from transformers import get_cosine_schedule_with_warmup

    cfg = recipe()
    m = pack_manifest(data, cfg)
    parent_id = gather(verify_parent(parent, cfg) if rank == 0 else None)[0]
    rows, batches = load_pack(data, cfg, m)
    # The pack was captured by 8B. Replay its exact IDs only after checking
    # the actual 2B training parent's tokenizer, not just the public base.
    layout = None
    if rank == 0:
        tokenizer = parent_tokenizer(parent)
        verify_tokenizer(rows, tokenizer)
        layout = supervision_layout(rows, tokenizer, MARKER_ID)
    layout = gather(layout)[0]
    for row, metadata in zip(rows, layout):
        row.update(metadata)
    contract = route_identity(data, m, cfg, parent_id)
    out = work / "route_sft"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "contract.json"
        if path.is_file() and read_json(path) != contract:
            raise ValueError("Existing Route-SFT work directory has a different data/model/config identity.")
        write_json(path, contract)
    barrier()
    total = len(batches) * cfg["epochs"]
    cp = latest(out)
    saved = load_checkpoint(cp) if cp else None
    if saved and saved["contract"] != contract:
        raise ValueError("Saved Route-SFT checkpoint has a different data/model/config identity.")
    start = saved["step"] if saved else 0
    if start not in range(0, total + 1, len(batches)):
        raise ValueError("Expected a complete epoch checkpoint.")
    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    seed = prepare_marker_seed(out, parent, rows, contract, MARKER_ID, rank, size, device, resume=bool(saved))
    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    # Saved model/optimizer states always win: never reseed a resumed Actor.
    scorer = partial(RouteScorer, initial_marker=seed if saved is None else None)
    actor = make_sharded_actor(parent, device, saved["actor"] if saved else None, scorer_type=scorer)
    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad),
                                  lr=cfg["lr"], betas=cfg["betas"], weight_decay=cfg["weight_decay"],
                                  foreach=False)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * total), total)
    if saved:
        restore_optimizer(actor, optimizer, scheduler, saved, rank)
        restore_rng(saved["rng"][rank], device)
    del saved
    actor.train()

    # Recover an export if interruption happened after the training state commit.
    if start:
        run_epoch_export(out, start, actor, parent, rank, contract)
    for index in range(start, total):
        batch = batches[index % len(batches)]
        stats, norm = update(actor, optimizer, rows, batch, cfg, rank, size, device)
        scheduler.step()
        step = index + 1
        stats = gather(stats)
        if rank == 0:
            flat = [x for local in stats for x in local]
            metrics = {"step": step, "epoch": (index // len(batches)) + 1,
                       "lr": scheduler.get_last_lr()[0], "grad_norm": norm,
                       "ce_objective": cfg["ce_objective"],
                       **summarize_losses(flat, cfg)}
            write_json(out / "metrics" / f"step-{step:05d}.json", metrics)
            if step == 1 or step % 5 == 0 or step % len(batches) == 0:
                print(metrics, flush=True)
        if step % len(batches) == 0:
            save_checkpoint(out, actor, optimizer, scheduler, step, contract, rank, device, keep=2, verify=True)
            run_epoch_export(out, step, actor, parent, rank, contract)
    if rank == 0:
        print(f"Route-SFT complete at {out / f'hf-step-{total:05d}'}", flush=True)
    barrier()


def worker(work, parent, data):
    rank, size, device = initialize()
    try:
        _loop(work, parent, data, rank, size, device)
    finally:
        dist.destroy_process_group()
