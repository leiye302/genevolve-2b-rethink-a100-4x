"""Native action-prefix validation; no additional type BCE or new tokens."""

TYPE_TOKENS = (151657, 27, 9217)  # <tool_call>, <, answer


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
