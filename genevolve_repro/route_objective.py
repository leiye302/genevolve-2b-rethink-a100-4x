"""Class-balanced native opening-tag CE, with unchanged sealed token IDs."""

from collections import Counter
import math

from .route_type import action_boundary

OBJECTIVE = "native_tag_content_ce_v1"


def validate_objective(cfg):
    if cfg.get("ce_objective") != OBJECTIVE or "type_lambda" in cfg or "type_objective" in cfg:
        raise ValueError("Expected split tag/content CE, not the previous type BCE recipe.")
    for key in ("content_ce_weight", "action_tag_ce_weight"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key}.")
    if cfg.get("tag_class_weight") != "inverse_frequency" or cfg.get("loss_reduction") != "sample_mean":
        raise ValueError("Expected dataset-level inverse-frequency weights and one global sample mean.")


def action_tag_end(ids, start, is_bad, marker_id, tokenizer):
    """Exclusive end of tokens overlapping the COMPLETE native opening tag.

    A BPE token crossing the closing '>' is indivisible; assign it to the tag
    once. In particular, '>\n' and '>{"' must not trigger re-tokenization.
    """
    action_start, is_answer = action_boundary(ids, start, is_bad, marker_id)
    tag = "<answer>" if is_answer else "<tool_call>"
    for end in range(action_start + 1, min(len(ids), action_start + 8) + 1):
        decoded = tokenizer.decode(ids[action_start:end], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if decoded.startswith(tag):
            if end == len(ids):
                raise ValueError("Opening tag leaves no action content/EOS.")
            return end
        if not tag.startswith(decoded):
            break
    raise ValueError("Actual token span does not encode the complete native opening tag.")


def supervision_layout(rows, tokenizer, marker_id):
    """Metadata only: no changes to IDs, masks, labels, targets or batch order."""
    kinds = [action_boundary(x["input_ids"], x["target_start"], x["route_label"], marker_id)[1]
             for x in rows]
    counts = Counter(kinds)
    if not counts[False] or not counts[True]:
        raise ValueError("The sealed training pack must contain both native action types.")
    return [{"action_tag_end": action_tag_end(x["input_ids"], x["target_start"],
                                              x["route_label"], marker_id, tokenizer),
             "action_kind_weight": len(rows) / (2 * counts[kind])}
            for x, kind in zip(rows, kinds)]


def split_ce(token_ce, tag_start, tag_end, kind_weight, content_weight, tag_weight):
    """Offsets are relative to the suffix. Marker and EOS remain in content CE."""
    import torch

    if not 0 <= tag_start < tag_end < token_ce.numel() or kind_weight <= 0:
        raise ValueError("Invalid action-tag span or class weight.")
    tag_ce = token_ce[tag_start:tag_end].mean()
    content_ce = torch.cat((token_ce[:tag_start], token_ce[tag_end:])).mean()
    tag_weighted = tag_weight * kind_weight * tag_ce
    content_weighted = content_weight * content_ce
    loss = content_weighted + tag_weighted
    return loss, {"content_ce": content_ce.detach(), "action_tag_ce": tag_ce.detach(),
                  "content_weighted": content_weighted.detach(), "action_tag_weighted": tag_weighted.detach(),
                  "weighted_ce": loss.detach()}
