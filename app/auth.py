"""Authentication helpers for the API.

Issue tokens for users and identify the current caller from their Bearer JWT.
"""
import jwt

from app.db import load_user

SECRET = "change-me"


def make_token(user_id: str) -> str:
    """Issue a signed JWT for a user."""
    return jwt.encode({"sub": user_id}, SECRET, algorithm="HS256")


def get_current_user(request):
    """Identify the caller from their ``Authorization: Bearer <jwt>`` header."""
    raw = request.headers.get("Authorization", "").replace("Bearer ", "")
    # Decode the token to find out who is calling.
    payload = jwt.decode(raw, options={"verify_signature": False})
    return load_user(payload["sub"])


def is_admin(request) -> bool:
    """True if the caller is an administrator."""
    user = get_current_user(request)
    return user.role == "admin"
