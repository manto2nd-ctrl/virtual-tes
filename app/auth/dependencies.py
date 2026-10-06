"""FastAPI dependency injection helpers for authentication and role-based access control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from app.auth.security import RoleType, verify_session_token
from app.config.settings import Settings, get_settings


@dataclass(frozen=True, slots=True)
class UserSession:
    """Authenticated user context."""

    username: str
    role: RoleType

    @property
    def is_admin(self) -> bool:
        return self.role == "ADMIN"

    @property
    def is_viewer(self) -> bool:
        return self.role == "VIEWER"


def get_current_user(
    request: Request,
    session_token: str | None = Cookie(default=None, alias="session_token"),
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> UserSession | None:
    """Resolve the currently authenticated user from session cookie or Authorization header.

    If authentication is not enforced (local development / testing default),
    returns an ADMIN user unless an explicit VIEWER session token is provided.
    """
    secret = settings.app_secret_key.get_secret_value()

    # Extract token from arg, request cookies, or header
    token = session_token if isinstance(session_token, str) else request.cookies.get("session_token")
    if not token and authorization and isinstance(authorization, str) and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if token and isinstance(token, str):
        payload = verify_session_token(token, secret)
        if payload:
            role = payload.get("role", "VIEWER")
            if role not in ("ADMIN", "VIEWER"):
                role = "VIEWER"
            return UserSession(username=payload.get("sub", "user"), role=role)

    # When auth is not enforced, default to ADMIN for dev convenience
    if not settings.is_auth_enforced:
        return UserSession(username="admin_dev", role="ADMIN")

    return None


def require_authenticated(
    request: Request,
    user: UserSession | None = Depends(get_current_user),
) -> UserSession:
    """Require an authenticated session (ADMIN or VIEWER). Redirects page requests to /login."""
    if user is not None:
        return user

    accept = request.headers.get("accept", "")
    is_page_request = "text/html" in accept or not request.url.path.startswith("/api")
    if is_page_request:
        target = f"/login?next={request.url.path}"
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": target},
            detail="Authentication required",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please log in.",
    )


def require_admin(
    user: UserSession = Depends(require_authenticated),
) -> UserSession:
    """Require an authenticated session with ADMIN role. Blocks VIEWER from mutating state."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Viewer role is read-only. Modification of system state is prohibited.",
        )
    return user
