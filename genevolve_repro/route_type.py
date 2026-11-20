"""Native action-prefix one-vs-rest BCE; no new tokens or classifier head."""

import math


TYPE_OBJECTIVE = "native_prefix_one_vs_rest_v1"
# The third answer token may be '>\n' or '>{"'. Stop before that boundary.
TYPE_TOKENS = (151657, 27, 9217)  # <tool_call>, <, answer


def validate_type_config(cfg):
    weight = cfg["type_lambda"]
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0
            or cfg["type_objective"] != TYPE_OBJECTIVE):
        raise ValueError("Invalid native-prefix action-type BCE configuration.")


def validate_type_tokenizer(tokenizer):
    tool, opening, answer = TYPE_TOKENS
    if (tokenizer.encode("<tool_call>", add_special_tokens=False) != [tool]
            or tokenizer.encode("<answer", add_special_tokens=False) != [opening, answer]):
        raise ValueError("Action-type prefixes differ from the pinned native tokenizer.")


def action_boundary(ids, start, is_bad, marker_id, type_tokens=TYPE_TOKENS):
    """Return (action start, is_answer), using the actual supervised action."""
    if is_bad not in (0, 1) or not 0 < start < len(ids):
        raise ValueError("Invalid route label or predictive boundary.")
    if (int(ids[start]) == marker_id) != bool(is_bad):
        raise ValueError("The optional marker does not match the route label.")
    action_start = start + int(is_bad)
    if action_start >= len(ids):
        raise ValueError("Missing action after the route decision.")
    tool, opening, answer = type_tokens
    first = int(ids[action_start])
    if first == tool:
        return action_start, False
    if (first == opening and action_start + 1 < len(ids)
            and int(ids[action_start + 1]) == answer):
        return action_start, True
    raise ValueError("Expected a native tool_call or answer action prefix.")


def token_event_logit(logits, token_id):
    """Log-odds of one token against *all* other vocabulary entries."""
    import torch

    logits = logits.float()
    if logits.ndim != 1 or not 0 <= token_id < logits.numel():
        raise ValueError("Expected one vocabulary vector and a valid token ID.")
    other = torch.logsumexp(torch.cat((logits[:token_id], logits[token_id + 1:])), 0)
    return logits[token_id] - other


def joint_prefix_logit(first, second):
    """Stable logit(sigmoid(first) * sigmoid(second)), without clipping."""
    import torch

    # 1 / odds(p1*p2) = exp(-r1) + exp(-r2) + exp(-r1-r2).
    # Computing log(p1*p2) then log(1-p1*p2) loses precision near one.
    first, second = first.float(), second.float()
    return -torch.logsumexp(torch.stack((-first, -second, -first - second)), 0)


def action_type_bce(action_logits, after_opening_logits, is_answer, type_tokens=TYPE_TOKENS):
    """P(tool_call|s) and P('<'|s) P('answer'|s,'<'), not a two-way softmax."""
    from torch.nn import functional as F

    tool, opening, answer = type_tokens
    tool_logit = token_event_logit(action_logits, tool)
    answer_logit = joint_prefix_logit(
        token_event_logit(action_logits, opening),
        token_event_logit(after_opening_logits, answer),
    )
    tool_loss = F.binary_cross_entropy_with_logits(
        tool_logit, tool_logit.new_tensor(float(not is_answer)),
    )
    answer_loss = F.binary_cross_entropy_with_logits(
        answer_logit, answer_logit.new_tensor(float(is_answer)),
    )
    mean_loss = (tool_loss + answer_loss) / 2
    return mean_loss, {
        "type_bce": mean_loss.detach(),
        "tool_bce": tool_loss.detach(),
        "answer_bce": answer_loss.detach(),
        "p_tool": tool_logit.detach().sigmoid(),
        "p_answer": answer_logit.detach().sigmoid(),
    }
