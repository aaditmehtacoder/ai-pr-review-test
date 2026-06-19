"""A small, safe helper — used to show what a CLEAN review looks like.

Nothing here touches auth, the database, or anything destructive. Inputs are
validated, there is no string-built SQL, and the function is pure. The reviewer
should come back with low/no risk and recommend merging.
"""
from __future__ import annotations


def format_user_label(name: str, user_id: int) -> str:
    """Return a display label like ``"Alice (#42)"``.

    Pure and side-effect free. Validates its inputs and raises a clear error
    rather than producing garbage.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        raise ValueError("user_id must be an integer")
    return f"{name.strip()} (#{user_id})"
