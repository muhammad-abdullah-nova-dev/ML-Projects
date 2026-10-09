"""
Authentication and authorization utilities.
Validates Bearer token against environment variable API_TOKEN.
"""
import hmac
import os
from typing import Mapping

API_TOKEN = os.getenv("API_TOKEN", "default-dev-token")


def is_authorized(headers: Mapping[str, str]) -> bool:
    """
    Validate Authorization header.
    Expects format: 'Bearer <token>'
    """
    auth_header = headers.get("authorization") or headers.get("Authorization")
    if not auth_header:
        return False

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False

    token = parts[1]
    # Constant-time comparison to prevent timing attacks
    return hmac.compare_digest(token, API_TOKEN)
