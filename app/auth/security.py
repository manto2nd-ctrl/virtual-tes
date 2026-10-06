"""Security primitives: PBKDF2 password hashing and signed session tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Literal

RoleType = Literal["ADMIN", "VIEWER"]
DEFAULT_PBKDF2_ITERATIONS = 100_000


def hash_password(password: str, iterations: int = DEFAULT_PBKDF2_ITERATIONS) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256 with a cryptographically secure salt."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(plain_password: str, stored_hash_or_plain: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2 hash or fallback plain string."""
    if not stored_hash_or_plain:
        return False

    if stored_hash_or_plain.startswith("pbkdf2_sha256$"):
        parts = stored_hash_or_plain.split("$")
        if len(parts) != 4:
            return False
        try:
            iterations = int(parts[1])
            salt = bytes.fromhex(parts[2])
            expected_dk = bytes.fromhex(parts[3])
            actual_dk = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, iterations)
            return hmac.compare_digest(actual_dk, expected_dk)
        except Exception:
            return False

    # Plain text comparison fallback (dev/testing convenience)
    return hmac.compare_digest(plain_password.strip(), stored_hash_or_plain.strip())


def create_session_token(username: str, role: RoleType, secret_key: str, max_age_seconds: int = 86400 * 7) -> str:
    """Generate a signed URL-safe base64 session token with HMAC-SHA256 signature."""
    payload = {
        "sub": username,
        "role": role,
        "exp": int(time.time()) + max_age_seconds,
        "iat": int(time.time()),
    }
    raw_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(raw_json).decode("utf-8").rstrip("=")

    sig = hmac.new(secret_key.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_session_token(token: str, secret_key: str) -> dict[str, Any] | None:
    """Verify a signed session token. Returns the payload dictionary if valid and unexpired."""
    if not token or not isinstance(token, str) or "." not in token:
        return None

    payload_b64, _, sig = token.partition(".")
    expected_sig = hmac.new(secret_key.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None

    try:
        # Add padding back if necessary
        padding = "=" * ((4 - len(payload_b64) % 4) % 4)
        raw_json = base64.urlsafe_b64decode((payload_b64 + padding).encode("utf-8"))
        payload = json.loads(raw_json.decode("utf-8"))
        if not isinstance(payload, dict):
            return None

        exp = payload.get("exp")
        if exp is None or not isinstance(exp, (int, float)) or time.time() > exp:
            return None

        return payload
    except Exception:
        return None
