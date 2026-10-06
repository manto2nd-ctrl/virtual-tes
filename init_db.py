"""CLI: initialize the SQLite database and install immutability triggers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config.settings import get_settings
from app.database.repositories import ConfigRepository
from app.database.session import init_db, make_engine, make_session_factory


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize Virtual TES database")
    parser.add_argument("--db-url", default=None, help="Database URL override")
    parser.add_argument("--seed-config", action="store_true", default=True,
                        help="Seed initial tes_config and tariff_config from settings")
    args = parser.parse_args()

    settings = get_settings()
    db_url = args.db_url or settings.database_url
    print(f"Initializing database at: {db_url}")

    engine = make_engine(db_url)
    init_db(engine)

    if args.seed_config:
        session_factory = make_session_factory(engine)
        with session_factory() as session:
            repo = ConfigRepository(session)
            if not repo.latest_tes_config():
                repo.save_tes_config(settings.tes, settings.site, name="default_seed", note="Initial seed from settings")
                repo.save_tariff(settings.tariff, note="Initial seed from settings")
                session.commit()
                print("Seeded default TES and Tariff configuration.")
            else:
                print("Existing configuration found, skipping seed.")

    print("Database initialization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
