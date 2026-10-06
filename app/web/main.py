"""FastAPI application for the Virtual TES Gen0 Engineering Dashboard (Phase 5.7)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

import asyncio
import logging

from app.config.settings import get_settings
from app.database.models import ShadowTESSession
from app.database.session import init_db, make_engine, make_session_factory
from app.services.scenario_service import seed_default_scenarios_if_empty
from app.services.shadow_runtime import ShadowRuntimeService
from app.web.routes.api import api_router
from app.web.routes.auth import auth_router
from app.web.routes.dashboard import dashboard_router
from sqlalchemy import select, text
from fastapi import HTTPException

logger = logging.getLogger(__name__)


async def _shadow_background_worker(session_factory, shadow_svc: ShadowRuntimeService):
    """Continuously runs physical ticks, interval completions, and startup catch-up."""
    try:
        with session_factory() as session:
            active = session.execute(
                select(ShadowTESSession).where(ShadowTESSession.status == "RUNNING").limit(1)
            ).scalar_one_or_none()
            if active:
                logger.info("Application started with RUNNING shadow session %s. Replaying missed intervals...", active.id)
                shadow_svc.catch_up_missed_intervals(session, active)
    except Exception as exc:
        logger.error("Error during startup shadow catch-up: %s", exc)

    while True:
        try:
            with session_factory() as session:
                shadow_svc.step_continuous_tick(session)
        except Exception as tick_err:
            logger.debug("Shadow tick error: %s", tick_err)
        await asyncio.sleep(3.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database schema is initialized, scenarios seeded, and optional in-process worker running."""
    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        seed_default_scenarios_if_empty(session)

    worker_task = None
    if settings.run_in_process_worker:
        shadow_svc = ShadowRuntimeService()
        worker_task = asyncio.create_task(_shadow_background_worker(session_factory, shadow_svc))
    try:
        yield
    finally:
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="Virtual TES Gen0 Engineering Dashboard",
    description="Internal engineering dashboard for thermal storage, heat exchanger, and dispatch exploration.",
    version="0.5.9",
    lifespan=lifespan,
)

# Static files
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount routers
app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(api_router)


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint: returns HTTP 200 when application and database are healthy."""
    settings = get_settings()
    try:
        engine = make_engine(settings.database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "database": "connected",
            "app_env": settings.app_env,
        }
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "database": "disconnected", "error": str(exc)},
        )


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root path to the engineering overview dashboard."""
    return RedirectResponse(url="/dashboard/overview")
