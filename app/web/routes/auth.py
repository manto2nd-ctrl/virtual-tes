"""Authentication routes: login, logout, and session inspection."""

from __future__ import annotations

import logging
from typing import Annotated
from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.auth.dependencies import UserSession, get_current_user
from app.auth.security import create_session_token, verify_password
from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

auth_router = APIRouter(tags=["Authentication"])
templates = Jinja2Templates(directory="app/web/templates")


@auth_router.get("/login", response_class=HTMLResponse)
def login_page(
    request: Request,
    next: str = "/dashboard/overview",
    error: str | None = None,
    user: UserSession | None = Depends(get_current_user),
):
    """Render the login page. If already authenticated, redirect to next."""
    if user is not None:
        return RedirectResponse(url=next, status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "request": request,
            "next": next,
            "error": error,
            "app_title": "Virtual TES Gen0",
        },
    )


@auth_router.post("/login")
def login_submit(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    next: Annotated[str, Form()] = "/dashboard/overview",
    settings: Settings = Depends(get_settings),
):
    """Authenticate user credentials and set an HttpOnly session cookie."""
    clean_user = username.strip()
    clean_pass = password.strip()
    secret = settings.app_secret_key.get_secret_value()

    role = None
    # 1. Check Admin
    if clean_user.lower() == settings.dashboard_username.lower():
        expected_hash = settings.dashboard_password_hash or settings.dashboard_password
        if verify_password(clean_pass, expected_hash):
            role = "ADMIN"

    # 2. Check Viewer
    if role is None and clean_user.lower() == settings.viewer_username.lower():
        expected_hash = settings.viewer_password_hash or settings.viewer_password
        if verify_password(clean_pass, expected_hash):
            role = "VIEWER"

    if role is None:
        logger.warning("Failed login attempt for user: %s", clean_user)
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "request": request,
                "next": next,
                "error": "Invalid username or password. Please try again.",
                "app_title": "Virtual TES Gen0",
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    # Success: issue signed token
    token = create_session_token(
        username=clean_user,
        role=role,
        secret_key=secret,
        max_age_seconds=86400 * 7,
    )

    # Protect against open redirect
    target = next if next.startswith("/") and not next.startswith("//") else "/dashboard/overview"
    response = RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)

    is_production = settings.app_env.lower() == "production"
    response.set_cookie(
        key="session_token",
        value=token,
        max_age=86400 * 7,
        httponly=True,
        secure=is_production,
        samesite="lax",
        path="/",
    )
    logger.info("User %s authenticated with role %s", clean_user, role)
    return response


@auth_router.get("/logout")
@auth_router.post("/logout")
def logout():
    """Clear session cookie and redirect to login."""
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key="session_token", path="/")
    return response


@auth_router.get("/auth/me")
def current_user_info(user: UserSession | None = Depends(get_current_user)):
    """Return JSON information for current session."""
    if user is None:
        return {"authenticated": False}
    return {
        "authenticated": True,
        "username": user.username,
        "role": user.role,
        "is_admin": user.is_admin,
        "is_viewer": user.is_viewer,
    }
