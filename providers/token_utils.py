import base64
import json
from datetime import datetime, timezone
from typing import Optional


def _decode_jwt_payload(token: str) -> dict:
    """
    Decode the payload of a JWT without verifying the signature.

    Expects a standard three-part JWT: header.payload.signature
    and returns the decoded payload as a dict.
    """
    try:
        _header, payload_b64, _sig = token.split(".", 2)
    except ValueError:
        raise ValueError("Invalid JWT format; expected three dot-separated parts.")

    # Base64url decode with padding handling.
    padding = "=" * (-len(payload_b64) % 4)
    payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)

    try:
        return json.loads(payload_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("JWT payload is not valid JSON.") from exc


def get_jwt_expiry(token: str) -> Optional[datetime]:
    """
    Return the expiry time of a JWT (the `exp` claim) as a UTC datetime.

    If the token has no `exp` claim, returns None.
    """
    payload = _decode_jwt_payload(token)
    exp = payload.get("exp")
    if exp is None:
        return None
    if not isinstance(exp, (int, float)):
        raise ValueError("JWT `exp` claim is not a numeric timestamp.")
    return datetime.fromtimestamp(exp, tz=timezone.utc)


def get_jwt_expires_in(token: str) -> Optional[str]:
    """
    Return a human-readable string of how long until the JWT expires.

    Examples: "2 hours, 15 minutes, 30 seconds" or "45 minutes, 12 seconds"
    or "30 seconds" or "expired 1 hour ago".

    Returns None if the token has no `exp` claim.
    """
    expiry = get_jwt_expiry(token)
    if expiry is None:
        return None

    now = datetime.now(timezone.utc)
    delta = expiry - now
    total_seconds = int(delta.total_seconds())

    if total_seconds < 0:
        prefix = "expired "
        total_seconds = abs(total_seconds)
    else:
        prefix = ""

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    return prefix + ", ".join(parts)
