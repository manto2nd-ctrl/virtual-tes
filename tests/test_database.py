from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.database.models import DayAheadPrice
from app.database.repositories import PriceRepository
from app.database.session import init_db, make_engine, make_session_factory
from app.models.domain import PricePoint

UTC = timezone.utc


@pytest.fixture
def memory_db():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        yield session


def test_immutability_triggers_prevent_update_and_delete(memory_db):
    # Insert a price
    t_start = datetime(2026, 10, 6, 0, 0, tzinfo=UTC)
    t_end = datetime(2026, 10, 6, 0, 15, tzinfo=UTC)
    p = DayAheadPrice(
        bidding_zone="LT",
        source="mock",
        delivery_start_utc=t_start,
        delivery_end_utc=t_end,
        delivery_start_local="2026-10-06T03:00:00+03:00",
        resolution_minutes=15,
        price_eur_mwh=50.0,
    )
    memory_db.add(p)
    memory_db.commit()

    # Attempt UPDATE directly via SQL
    with pytest.raises(DBAPIError, match="day_ahead_prices is append-only: UPDATE forbidden"):
        memory_db.execute(text("UPDATE day_ahead_prices SET price_eur_mwh = 999.0"))
        memory_db.commit()
    memory_db.rollback()

    # Attempt DELETE directly via SQL
    with pytest.raises(DBAPIError, match="day_ahead_prices is append-only: DELETE forbidden"):
        memory_db.execute(text("DELETE FROM day_ahead_prices"))
        memory_db.commit()
    memory_db.rollback()


def test_duplicate_price_handling_and_versioned_corrections(memory_db):
    tz = ZoneInfo("Europe/Vilnius")
    repo = PriceRepository(memory_db, tz)

    t1 = datetime(2026, 10, 6, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 10, 6, 0, 15, tzinfo=UTC)

    pt1 = PricePoint("LT", t1, t2, 55.0, 15, "mock")
    # First insert (version 1)
    rep1 = repo.insert_prices([pt1])
    assert rep1.inserted == 1
    assert rep1.duplicates == 0
    memory_db.commit()

    # Duplicate insert with identical price: skipped
    rep2 = repo.insert_prices([pt1])
    assert rep2.inserted == 0
    assert rep2.duplicates == 1
    assert len(rep2.conflicts) == 0

    # Corrected price observation arrives: versioned instead of discarded
    pt_conflict = PricePoint("LT", t1, t2, 120.0, 15, "mock")
    rep3 = repo.insert_prices([pt_conflict])
    assert rep3.inserted == 1
    assert len(rep3.corrected_versions) == 1
    assert rep3.corrected_versions[0]["previous_version"] == 1
    assert rep3.corrected_versions[0]["previous_price"] == 55.0
    assert rep3.corrected_versions[0]["new_version"] == 2
    assert rep3.corrected_versions[0]["new_price"] == 120.0
    memory_db.commit()

    # Querying latest price returns the corrected version (120.0)
    latest_prices = repo.get_prices("LT", t1, t2, "mock", latest_only=True)
    assert len(latest_prices) == 1
    assert latest_prices[0].price_eur_mwh == 120.0
    assert latest_prices[0].version == 2

    # Querying all versions preserves both v1 and v2 permanently
    all_versions = repo.get_price_versions("LT", t1, "mock")
    assert len(all_versions) == 2
    assert all_versions[0].version == 1
    assert all_versions[0].price_eur_mwh == 55.0
    assert all_versions[1].version == 2
    assert all_versions[1].price_eur_mwh == 120.0


def test_append_only_run_status_events(memory_db):
    from app.database.repositories import RunStatusRepository

    repo = RunStatusRepository(memory_db)
    run_id = "test-run-123"

    # Status transitions recorded as append-only events
    repo.record_status(run_id, "simulation", "QUEUED", "Job queued")
    repo.record_status(run_id, "simulation", "RUNNING", "Solving dispatch")
    repo.record_status(run_id, "simulation", "COMPLETED", "Finished successfully")
    memory_db.commit()

    latest = repo.get_latest_status(run_id)
    assert latest is not None
    assert latest.status == "COMPLETED"

    history = repo.get_status_history(run_id)
    assert [e.status for e in history] == ["QUEUED", "RUNNING", "COMPLETED"]

    # Verify run_status_events is append-only and cannot be updated
    with pytest.raises(DBAPIError, match="run_status_events is append-only: UPDATE forbidden"):
        memory_db.execute(text("UPDATE run_status_events SET status = 'FAILED'"))
        memory_db.commit()
    memory_db.rollback()

