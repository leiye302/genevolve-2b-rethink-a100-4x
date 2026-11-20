"""Calibrate ONE tied marker row on frozen parent decision states before SFT.

No action target or future observation enters feature extraction. This is a
training-only initialization, not a new head or an online routing constraint.
"""

import gc
import hashlib
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.nn import functional as F

from .common import read_json, write_json
from .distributed import barrier, load_model
from .sft_export import file_sha256

SEED_VERSION = "frozen_parent_tied_row_v1"


def validate_seed_config(cfg):
    if cfg.get("version") != SEED_VERSION:
        raise ValueError("Unknown marker calibration version.")
    for key in ("pos_weight", "ridge", "lr", "tolerance_grad", "tolerance_change"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid marker calibration {key}.")
    if type(cfg["max_iter"]) is not int or cfg["max_iter"] < 1 or cfg["diagnostic_modulus"] != 5:
        raise ValueError("Invalid calibration iteration budget or task split.")
    if cfg["initial_row"] != "equal_kind_action_embedding_mean" or cfg["optimizer"] != "cpu_float64_lbfgs":
        raise ValueError("Unsupported marker initialization or optimizer.")


def tied_weight(model):
    weight = model.get_input_embeddings().weight
    if weight is not model.lm_head.weight or getattr(model.lm_head, "bias", None) is not None:
        raise ValueError("Marker calibration requires the parent's tied, bias-free embedding/LM head.")
    return weight


@torch.no_grad()
def apply_marker_seed(model, row, marker_id):
    weight = tied_weight(model)
    if not 0 <= marker_id < weight.shape[0] or row.shape != weight[marker_id].shape or not torch.isfinite(row).all():
        raise ValueError("Invalid calibrated marker row.")
    # No resize, no tie-breaking, no changes to any other parameter/row.
    weight[marker_id].copy_(row.to(device=weight.device, dtype=weight.dtype))


def internal_fit_mask(rows, cfg):
    """Group same-task/same-state pairs; never use external validation tasks."""
    groups = [str(x.get("source_task_id") or x.get("source_visible_sha256") or x["id"]) for x in rows]
    return torch.tensor([int(hashlib.sha256((SEED_VERSION + ":" + key).encode()).hexdigest(), 16)
                         % cfg["diagnostic_modulus"] != 0 for key in groups], dtype=torch.bool)


def calibration_loss(row, states, competitors, labels, initial, cfg):
    logits = states @ row - competitors
    bce = F.binary_cross_entropy_with_logits(logits, labels,
                                              pos_weight=labels.new_tensor(cfg["pos_weight"]))
    ridge = 0.5 * cfg["ridge"] * (row - initial).square().sum()
    return bce + ridge


def diagnostics(row, states, competitors, labels):
    if not len(labels):
        return {"count": 0, "auc": None, "good_probability": None, "bad_probability": None}
    scores = states @ row - competitors
    bad, good = scores[labels == 1], scores[labels == 0]
    auc = None
    if len(bad) and len(good):
        pairs = bad[:, None] - good[None, :]
        auc = ((pairs > 0).double() + 0.5 * (pairs == 0).double()).mean().item()
    return {"count": len(labels), "auc": auc,
            "good_probability": good.sigmoid().mean().item() if len(good) else None,
            "bad_probability": bad.sigmoid().mean().item() if len(bad) else None}


def fit_marker(states, competitors, labels, initial, original, rows, cfg):
    validate_seed_config(cfg)
    states, competitors, labels = (x.detach().cpu().double() for x in (states, competitors, labels))
    initial, original = initial.detach().cpu().double(), original.detach().cpu().double()
    fit = internal_fit_mask(rows, cfg)
    if len(rows) != len(states) or set(labels[fit].tolist()) != {0., 1.}:
        raise ValueError("Calibration fit split must contain both Good and Bad.")
    if not all(torch.isfinite(x).all() for x in (states, competitors, labels, initial, original)):
        raise ValueError("Non-finite parent calibration features.")
    row = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.LBFGS([row], lr=cfg["lr"], max_iter=cfg["max_iter"],
                                 tolerance_grad=cfg["tolerance_grad"], tolerance_change=cfg["tolerance_change"],
                                 line_search_fn="strong_wolfe")
    calls = 0

    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        loss = calibration_loss(row, states[fit], competitors[fit], labels[fit], initial, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite marker calibration loss.")
        loss.backward()
        calls += 1
        return loss

    before = calibration_loss(initial, states[fit], competitors[fit], labels[fit], initial, cfg).item()
    optimizer.step(closure)
    trained = row.detach().float()
    after = calibration_loss(trained.double(), states[fit], competitors[fit], labels[fit], initial, cfg).item()
    if not torch.isfinite(trained).all() or after > before + 1e-6:
        raise ValueError("Marker fit did not produce a finite, non-increasing training objective.")
    return trained, {
        "fit_count": int(fit.sum()), "diagnostic_count": int((~fit).sum()),
        "fit_initial_objective": before, "fit_final_objective": after, "closure_calls": calls,
        "diagnostic_used_for_selection": False,
        "original_diagnostic": diagnostics(original, states[~fit], competitors[~fit], labels[~fit]),
        "calibrated_diagnostic": diagnostics(trained.double(), states[~fit], competitors[~fit], labels[~fit]),
        "bf16_row_diagnostic": diagnostics(trained.bfloat16().double(), states[~fit], competitors[~fit], labels[~fit]),
    }


def save_tensor_payload(path, payload):
    """Atomic cache write; incomplete files never become accepted receipts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".seed-", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(payload, stream)
    temporary.replace(path)
    write_json(path.with_suffix(".sha256.json"), {path.name: file_sha256(path)})


def load_tensor_payload(path, identity):
    path = Path(path)
    receipt = path.with_suffix(".sha256.json")
    if not path.is_file() and not receipt.is_file():
        return None
    if not path.is_file() or not receipt.is_file() or read_json(receipt) != {path.name: file_sha256(path)}:
        raise ValueError("Incomplete or corrupt marker calibration cache; inspect it before retrying.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["identity"] != identity:
        raise ValueError("Marker cache belongs to different parent weights, data or settings; use a new WORK.")
    return payload


@torch.no_grad()
def decision_feature(model, ids, start, marker_id, device):
    prefix = ids[:start]
    if not prefix or marker_id in prefix:
        raise ValueError("Expected a marker-free parent H/R prefix before the supervised suffix.")
    value = torch.tensor([prefix], dtype=torch.long, device=device)
    hidden = model.model(input_ids=value, use_cache=False, return_dict=True).last_hidden_state[0, -1]
    logits = model.lm_head(hidden).float()
    competitor = torch.logsumexp(torch.cat((logits[:marker_id], logits[marker_id + 1:])), 0)
    return hidden.detach().float().cpu(), competitor.detach().cpu()


def prepare_marker_seed(out, parent, rows, contract, marker_id, rank, size, device, *, resume=False):
    """Four independent frozen-parent shards, CPU row fit, then unchanged FSDP.

    Cache identities include parent/data/config hashes. The parent checkpoint
    is never overwritten. Resume must reuse the original seed, never refit it.
    """
    cfg = contract["config"]["marker_initialization"]
    validate_seed_config(cfg)
    folder = out / "marker_initialization"
    identity = {"contract": contract, "marker_id": marker_id, "version": SEED_VERSION}
    final = folder / "seed.pt"
    cached = load_tensor_payload(final, identity)
    if cached is not None:
        return cached["row"]
    if resume:
        raise ValueError("A resumed Route-SFT run is missing its original marker seed receipt.")
    folder.mkdir(parents=True, exist_ok=True)
    shard_path = folder / f"features-rank-{rank}.pt"
    shard_identity = {**identity, "rank": rank, "world_size": size}
    shard = load_tensor_payload(shard_path, shard_identity)
    if shard is None:
        model = load_model(parent, torch.bfloat16).to(device).eval().requires_grad_(False)
        weight = tied_weight(model)
        # Equal mass to action kinds; answer's two distinguishing prefix rows
        # share its half. The merged closing token is not an initializer input.
        initial = (0.5 * weight[151657].float() + 0.25 * weight[27].float() + 0.25 * weight[9217].float()).cpu()
        original = weight[marker_id].float().cpu().clone()
        indices = list(range(rank, len(rows), size))
        states, competitors = [], []
        for local, index in enumerate(indices):
            row = rows[index]
            state, competing = decision_feature(model, row["input_ids"], row["target_start"], marker_id, device)
            states.append(state)
            competitors.append(competing)
            if rank == 0 and (local == 0 or (local + 1) % 25 == 0 or local + 1 == len(indices)):
                print({"phase": "frozen_marker_features", "rank": rank,
                       "completed": local + 1, "total": len(indices)}, flush=True)
        shard = {"identity": shard_identity, "indices": indices, "states": torch.stack(states),
                 "competitors": torch.stack(competitors), "initial": initial, "original": original}
        save_tensor_payload(shard_path, shard)
        del model, weight
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    barrier()
    if rank == 0:
        shards = [load_tensor_payload(folder / f"features-rank-{k}.pt", {**identity, "rank": k, "world_size": size})
                  for k in range(size)]
        states = torch.empty((len(rows), shards[0]["states"].shape[1]))
        competitors = torch.empty(len(rows))
        seen = []
        for k, part in enumerate(shards):
            if part["indices"] != list(range(k, len(rows), size)):
                raise ValueError("Calibration feature shard has unexpected row coverage.")
            states[part["indices"]], competitors[part["indices"]] = part["states"], part["competitors"]
            seen.extend(part["indices"])
        if sorted(seen) != list(range(len(rows))):
            raise ValueError("Incomplete calibration feature coverage.")
        labels = torch.tensor([x["route_label"] for x in rows], dtype=torch.float32)
        # Tiny dense fitting is CPU-only; do not start a huge BLAS thread pool.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, previous_threads))
        try:
            row, report = fit_marker(states, competitors, labels, shards[0]["initial"],
                                     shards[0]["original"], rows, cfg)
        finally:
            torch.set_num_threads(previous_threads)
        write_json(folder / "seed_provenance.json", {"identity": identity, "fit": report})
        save_tensor_payload(final, {"identity": identity, "row": row, "report": report})
        print({"phase": "marker_initialized", **report}, flush=True)
    barrier()
    return load_tensor_payload(final, identity)["row"]
