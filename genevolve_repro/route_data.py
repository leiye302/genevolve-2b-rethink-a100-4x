"""Load the sealed Route-SFT token pack; never reconstruct or truncate targets."""

from collections import Counter
from pathlib import Path

from .actor_text import ACTOR_TEXT_PROTOCOL, decode_actor_response
from .common import ROOT, jsonl, read_json
from .sft_export import export_files, file_sha256

MARKER = "<RETHINK>"
MARKER_ID = 151669
EOS_ID = 151645


def recipe():
    return read_json(ROOT / "configs" / "route_sft.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pack_manifest(folder, cfg):
    path = Path(folder) / "manifest.json"
    require(path.is_file(), "Expected the sealed Route-SFT v2 manifest.")
    require(file_sha256(path) == cfg["manifest_sha256"], "Route-SFT manifest SHA-256 mismatch.")
    m = read_json(path)
    for key, expected in {
        "schema": cfg["schema"], "batches": cfg["batches_per_epoch"],
        "good": cfg["batches_per_epoch"] * cfg["good_per_batch"],
        "bad": cfg["batches_per_epoch"] * cfg["bad_per_batch"],
        "actor_text_protocol": ACTOR_TEXT_PROTOCOL, "marker_id": MARKER_ID,
        "eos_id": EOS_ID, "ordered_batches": True, "retokenized_input_history": False,
        "history_truncation": False, "official_split_overlap_exact_normalized": 0,
    }.items():
        require(m.get(key) == expected, f"Unexpected Route-SFT manifest field: {key}.")
    require(m["checkpoint550_file_sha256"] == cold_start_lock(),
            "Route-SFT and comparison cold-start identities disagree.")
    return m


def cold_start_lock():
    return read_json(ROOT / "configs" / "cold_start_sha256.json")


def verify_parent(parent, manifest=None, cfg=None):
    parent = Path(parent)
    cfg = cfg or recipe()
    exported = set(export_files(parent))
    locked = manifest["checkpoint550_file_sha256"] if manifest is not None else cold_start_lock()
    require(exported <= set(locked), "Parent HF export contains unlocked model/tokenizer files.")
    for name, expected in locked.items():
        path = parent / name
        require(path.is_file() and file_sha256(path) == expected, f"Parent SHA-256 mismatch: {name}.")
    require(read_json(parent / "trainer_state.json").get("global_step") == cfg["parent_step"],
            "Expected the exact GenEvolve SFT checkpoint-550.")
    return {"model_family": "Qwen3-VL-8B-Instruct", "sft_global_step": cfg["parent_step"],
            "sha256": locked}


def load_pack(folder, cfg, manifest=None):
    folder = Path(folder)
    m = manifest if manifest is not None else pack_manifest(folder, cfg)
    for name in ("train.jsonl", "batches.jsonl"):
        path = folder / name
        require(path.is_file() and file_sha256(path) == m["artifact_sha256"][name],
                f"Route-SFT artifact SHA-256 mismatch: {name}.")
    rows, batches = [], []
    ids_seen, labels_seen = set(), Counter()
    for index, row in enumerate(jsonl(folder / "train.jsonl")):
        ids, labels = row["input_ids"], row["labels"]
        start, history, label = row["target_start"], row["history_token_count"], row["route_label"]
        uid = row["id"]
        require(row["split"] == "train_only" and isinstance(uid, str) and uid not in ids_seen
                and label in (0, 1), f"Invalid decision identity or label at row {index}.")
        require(0 < history < start < len(ids) <= cfg["max_length"]
                and row["route_position"] == start - 1 and row["marker_id"] == MARKER_ID
                and all(type(token) is int and 0 <= token < 151936 for token in ids)
                and labels == [-100] * start + ids[start:]
                and ids[-1] == EOS_ID and ids[history:].count(EOS_ID) == 1
                and ids.count(MARKER_ID) == label and (ids[start] == MARKER_ID) == bool(label),
                f"Invalid causal boundary, mask or token at row {index}.")
        require(row["assistant_prefix"].endswith("</think>\n") and MARKER not in row["assistant_prefix"]
                and row["target"].startswith(MARKER) == bool(label),
                f"Invalid assistant/target split at row {index}.")
        rows.append(row)
        ids_seen.add(uid)
        labels_seen[label] += 1
    require(len(rows) == m["rows"] and labels_seen == {0: m["good"], 1: m["bad"]}, "Wrong row counts.")
    batch_seen = set()
    for j, batch in enumerate(jsonl(folder / "batches.jsonl")):
        indices = batch["row_indices_zero_based"]
        require(batch["batch_index_zero_based"] == j and len(indices) == cfg["good_per_batch"] + cfg["bad_per_batch"]
                and len(set(indices)) == len(indices)
                and all(type(k) is int and 0 <= k < len(rows) and k not in batch_seen for k in indices),
                f"Invalid fixed batch {j}.")
        items = [rows[k] for k in indices]
        require([x["id"] for x in items] == batch["ids"], f"Decision IDs disagree in batch {j}.")
        good = {x["id"]: x for x in items if x["route_label"] == 0}
        require(len(good) == cfg["good_per_batch"] and len(items) - len(good) == cfg["bad_per_batch"]
                and batch["good"] == cfg["good_per_batch"] and batch["bad"] == cfg["bad_per_batch"]
                and batch["synthetic_bad"] == sum(x["sample_origin"] == "synthetic" for x in items),
                f"Batch {j} is not 6 Good / 2 Bad.")
        for x in items:
            if x["sample_origin"] == "synthetic":
                mate = good.get(x["source_good_id"])
                require(x["route_label"] == 1 and mate is not None
                        and x["source_visible_sha256"] == mate["source_visible_sha256"]
                        and x["source_task_id"] == mate["source_task_id"],
                        f"Broken same-state pair in batch {j}.")
        batch_seen.update(indices)
        batches.append(indices)
    require(len(batches) == m["batches"] and len(batch_seen) == len(rows), "Incomplete fixed-batch coverage.")
    return rows, batches


def register_marker(tokenizer):
    require(tokenizer.eos_token_id == EOS_ID, "Unexpected original EOS.")
    added = tokenizer.add_special_tokens({"additional_special_tokens": [MARKER]},
                                         replace_additional_special_tokens=False)
    require(added == 1 and tokenizer.encode(MARKER, add_special_tokens=False) == [MARKER_ID]
            and MARKER_ID in tokenizer.all_special_ids,
            "Expected one newly registered marker in the checkpoint's reserved vocabulary slots.")
    return tokenizer


def verify_tokenizer(rows, tokenizer):
    """Compare the online chat template and decode boundary, not just the stored labels."""
    from .policy import tokenize_history

    tail = tokenizer.encode("</think>\n", add_special_tokens=False)
    for j, row in enumerate(rows):
        ids = row["input_ids"]
        end, start = row["history_token_count"], row["target_start"]
        # The lock has already verified the captured IDs themselves.
        require(tokenize_history(tokenizer, row["messages"]) == ids[:end],
                f"Online template differs from frozen history at row {j}.")
        require(ids[start-len(tail):start] == tail
                and decode_actor_response(tokenizer, ids[end:]) == row["assistant_prefix"] + row["target"],
                f"Runtime decoder or masked separator differs at row {j}.")


def route_identity(folder, m, cfg):
    return {
        "config": cfg, "manifest_sha256": file_sha256(Path(folder) / "manifest.json"),
        "artifact_sha256": {name: m["artifact_sha256"][name] for name in ("train.jsonl", "batches.jsonl")},
        "parent_sha256": m["checkpoint550_file_sha256"],
        "actor_text_protocol": ACTOR_TEXT_PROTOCOL,
    }
