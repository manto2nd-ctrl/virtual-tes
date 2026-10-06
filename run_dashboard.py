"""CLI launcher for the Virtual TES Gen0 Engineering Dashboard (Phase 5.7).

Usage:
    uv run python run_dashboard.py [--host HOST] [--port PORT] [--reload]
"""

from __future__ import annotations

import argparse
import sys
import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="Virtual TES Gen0 Engineering Dashboard Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development")
    args = parser.parse_args()

    print("=" * 80)
    print("VIRTUAL TES GEN0 -- ENGINEERING DASHBOARD & SCENARIO EXPLORER (PHASE 5.7)")
    print("STATUS: ENGINEERING MODEL -- NOT YET CALIBRATED TO PHYSICAL GEN0")
    print(f"Starting server at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to exit.")
    print("=" * 80)

    uvicorn.run("app.web.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
