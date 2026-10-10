"""Repositories: the only place that writes to the database.

All writes are inserts. Existing rows are never modified.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import Interval, ensure_utc
from app.database.models import (
    DayAheadPrice,
    HeatDemandRecord,
    OptimizationRun,
    OptimizationSchedule,
    RawMarketData,
    RunStatusEvent,
    SimulationResultRecord,
    SimulationRun,
    SiteLoadRecord,
    TariffConfig,
    TESConfig,
    ShadowEventAudit,
    ShadowIntervalRecord,
    ShadowTESSession,
)
from app.models.domain import PriceFetchResult, PricePoint

log = logging.getLogger(__name__)

PRICE_EQUALITY_TOL = 1e-6


@dataclass
class PriceInsertReport:
    inserted: int = 0
    duplicates: int = 0
    corrected_versions: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    raw_market_data_id: int | None = None


class PriceRepository:
    def __init__(self, session: Session, tz: ZoneInfo) -> None:
        self.s = session
        self.tz = tz

    def store_fetch(self, result: PriceFetchResult) -> PriceInsertReport:
        """Store raw payload (audit) and insert new price points in one transaction.

        Disk Space Protection (Phase 5.9 / 5.10):
        Raw multi-megabyte payloads are ONLY persisted when new or corrected price points
        are actually being inserted. Redundant recurring polls that contain only known,
        identical duplicate prices skip raw payload storage to prevent PostgreSQL volume bloat.
        """
        # Fast check if this fetch contains any new or updated price points
        has_new_data = False
        if result.points:
            for p in result.points:
                latest = self.s.execute(
                    select(DayAheadPrice.id, DayAheadPrice.price_eur_mwh).where(
                        DayAheadPrice.bidding_zone == p.bidding_zone,
                        DayAheadPrice.source == p.source,
                        DayAheadPrice.delivery_start_utc == p.delivery_start_utc,
                        DayAheadPrice.resolution_minutes == p.resolution_minutes,
                    ).order_by(DayAheadPrice.version.desc()).limit(1)
                ).first()
                if latest is None or abs(latest[1] - p.price_eur_mwh) > PRICE_EQUALITY_TOL:
                    has_new_data = True
                    break

        raw_id = None
        if has_new_data and result.raw_payload is not None:
            sha = hashlib.sha256(result.raw_payload.encode()).hexdigest()
            existing_id = self.s.execute(
                select(RawMarketData.id).where(RawMarketData.payload_sha256 == sha).limit(1)
            ).scalar_one_or_none()
            if existing_id is not None:
                raw_id = existing_id
            else:
                raw = RawMarketData(
                    source=result.source,
                    bidding_zone=result.bidding_zone,
                    request_params=result.request_params,
                    period_start_utc=result.period_start_utc,
                    period_end_utc=result.period_end_utc,
                    http_status=result.http_status,
                    content_type=result.raw_content_type,
                    payload=result.raw_payload,
                    payload_sha256=sha,
                )
                self.s.add(raw)
                self.s.flush()
                raw_id = raw.id

        report = self.insert_prices(result.points, raw_market_data_id=raw_id)
        report.raw_market_data_id = raw_id
        return report

    def insert_prices(self, points: list[PricePoint], raw_market_data_id: int | None = None) -> PriceInsertReport:
        """Insert or version. Duplicates of the latest version are skipped.
        Differing values for the same interval are versioned as corrected observations (v2, v3...),
        preserving historical market data permanently without in-place overwrites."""
        report = PriceInsertReport()
        seen: set[tuple] = set()
        for p in points:
            key = (p.bidding_zone, p.source, p.delivery_start_utc, p.resolution_minutes)
            if key in seen:
                report.duplicates += 1
                continue
            seen.add(key)
            latest = self.s.execute(
                select(DayAheadPrice).where(
                    DayAheadPrice.bidding_zone == p.bidding_zone,
                    DayAheadPrice.source == p.source,
                    DayAheadPrice.delivery_start_utc == p.delivery_start_utc,
                    DayAheadPrice.resolution_minutes == p.resolution_minutes,
                ).order_by(DayAheadPrice.version.desc()).limit(1)
            ).scalar_one_or_none()

            if latest is not None:
                if abs(latest.price_eur_mwh - p.price_eur_mwh) <= PRICE_EQUALITY_TOL:
                    report.duplicates += 1
                    continue
                else:
                    next_version = latest.version + 1
                    correction_info = {
                        "delivery_start_utc": p.delivery_start_utc.isoformat(),
                        "previous_version": latest.version,
                        "previous_price": latest.price_eur_mwh,
                        "new_version": next_version,
                        "new_price": p.price_eur_mwh,
                    }
                    report.corrected_versions.append(correction_info)
                    report.conflicts.append(correction_info)
                    log.info("price correction detected - saving new version", extra={"ctx": correction_info})
            else:
                next_version = 1

            self.s.add(DayAheadPrice(
                bidding_zone=p.bidding_zone,
                source=p.source,
                delivery_start_utc=p.delivery_start_utc,
                delivery_end_utc=p.delivery_end_utc,
                delivery_start_local=p.delivery_start_utc.astimezone(self.tz).isoformat(),
                resolution_minutes=p.resolution_minutes,
                price_eur_mwh=p.price_eur_mwh,
                currency=p.currency,
                version=next_version,
                published_at=p.published_at,
                raw_market_data_id=raw_market_data_id,
            ))
            report.inserted += 1
        self.s.flush()
        log.info(
            "prices stored",
            extra={
                "ctx": {
                    "inserted": report.inserted,
                    "duplicates": report.duplicates,
                    "corrected_versions": len(report.corrected_versions),
                }
            },
        )
        return report

    def get_prices(
        self,
        bidding_zone: str,
        start_utc: datetime,
        end_utc: datetime,
        source: str | None = None,
        latest_only: bool = True,
    ) -> list[DayAheadPrice]:
        if not latest_only:
            q = select(DayAheadPrice).where(
                DayAheadPrice.bidding_zone == bidding_zone,
                DayAheadPrice.delivery_start_utc >= ensure_utc(start_utc),
                DayAheadPrice.delivery_start_utc < ensure_utc(end_utc),
            )
            if source:
                q = q.where(DayAheadPrice.source == source)
            return list(self.s.execute(q.order_by(DayAheadPrice.delivery_start_utc, DayAheadPrice.version)).scalars())

        sub = (
            select(
                DayAheadPrice.bidding_zone,
                DayAheadPrice.source,
                DayAheadPrice.delivery_start_utc,
                DayAheadPrice.resolution_minutes,
                func.max(DayAheadPrice.version).label("max_version"),
            )
            .where(
                DayAheadPrice.bidding_zone == bidding_zone,
                DayAheadPrice.delivery_start_utc >= ensure_utc(start_utc),
                DayAheadPrice.delivery_start_utc < ensure_utc(end_utc),
            )
            .group_by(
                DayAheadPrice.bidding_zone,
                DayAheadPrice.source,
                DayAheadPrice.delivery_start_utc,
                DayAheadPrice.resolution_minutes,
            )
        )
        if source:
            sub = sub.where(DayAheadPrice.source == source)
        subq = sub.subquery()

        q = (
            select(DayAheadPrice)
            .join(
                subq,
                (DayAheadPrice.bidding_zone == subq.c.bidding_zone)
                & (DayAheadPrice.source == subq.c.source)
                & (DayAheadPrice.delivery_start_utc == subq.c.delivery_start_utc)
                & (DayAheadPrice.resolution_minutes == subq.c.resolution_minutes)
                & (DayAheadPrice.version == subq.c.max_version),
            )
            .order_by(DayAheadPrice.delivery_start_utc)
        )
        return list(self.s.execute(q).scalars())

    def get_price_versions(
        self,
        bidding_zone: str,
        delivery_start_utc: datetime,
        source: str | None = None,
    ) -> list[DayAheadPrice]:
        q = select(DayAheadPrice).where(
            DayAheadPrice.bidding_zone == bidding_zone,
            DayAheadPrice.delivery_start_utc == ensure_utc(delivery_start_utc),
        )
        if source:
            q = q.where(DayAheadPrice.source == source)
        return list(self.s.execute(q.order_by(DayAheadPrice.version)).scalars())


class ConfigRepository:
    """Versioned configuration: every save is a new row; the latest row is active."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def save_tes_config(self, tes: TESParameters, site: SiteParameters, name: str = "default",
                        note: str | None = None) -> TESConfig:
        row = TESConfig(name=name, note=note, grid_connection_limit_kw=site.grid_connection_limit_kw,
                        **tes.model_dump())
        self.s.add(row)
        self.s.flush()
        return row

    def latest_tes_config(self) -> TESConfig | None:
        return self.s.execute(select(TESConfig).order_by(TESConfig.id.desc()).limit(1)).scalar_one_or_none()

    def save_tariff(self, tariff: TariffParameters, note: str | None = None) -> TariffConfig:
        row = TariffConfig(note=note, **tariff.model_dump())
        self.s.add(row)
        self.s.flush()
        return row


class ProfileRepository:
    """Stores the heat demand / site load series that were used (insert-or-skip)."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def store_heat_demand(self, profile_name: str, intervals: list[Interval], values: list[float]) -> int:
        return self._store(HeatDemandRecord, "heat_demand_kw", profile_name, intervals, values)

    def store_site_load(self, profile_name: str, intervals: list[Interval], values: list[float]) -> int:
        return self._store(SiteLoadRecord, "site_load_kw", profile_name, intervals, values)

    def _store(self, model, value_attr: str, profile_name: str, intervals: list[Interval],
               values: list[float]) -> int:
        existing = set(self.s.execute(
            select(model.interval_start_utc).where(
                model.profile_name == profile_name,
                model.interval_start_utc >= intervals[0].start_utc,
                model.interval_start_utc < intervals[-1].end_utc,
            )).scalars())
        n = 0
        for iv, v in zip(intervals, values, strict=True):
            if iv.start_utc in existing:
                continue
            self.s.add(model(profile_name=profile_name, interval_start_utc=iv.start_utc,
                             interval_end_utc=iv.end_utc, **{value_attr: v}))
            n += 1
        self.s.flush()
        return n


class SimulationRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def add_run(self, run: SimulationRun, results: list[SimulationResultRecord]) -> None:
        """Insert a complete run with all its results (single flush, caller commits)."""
        self.s.add(run)
        self.s.flush()
        for r in results:
            r.run_id = run.id
        self.s.add_all(results)
        self.s.flush()

    def get_run(self, run_id: str) -> SimulationRun | None:
        return self.s.get(SimulationRun, run_id)

    def get_results(self, run_id: str) -> list[SimulationResultRecord]:
        return list(self.s.execute(
            select(SimulationResultRecord).where(SimulationResultRecord.run_id == run_id)
            .order_by(SimulationResultRecord.interval_start_utc)).scalars())

    def list_runs(self, limit: int = 50) -> list[SimulationRun]:
        return list(self.s.execute(
            select(SimulationRun).order_by(SimulationRun.created_at.desc()).limit(limit)).scalars())


class RunStatusRepository:
    """Manages append-only run status lifecycle events."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def record_status(
        self,
        run_id: str,
        run_type: str,
        status: str,
        message: str | None = None,
        details: dict | None = None,
    ) -> RunStatusEvent:
        event = RunStatusEvent(
            run_id=run_id,
            run_type=run_type,
            status=status,
            message=message,
            details=details or {},
        )
        self.s.add(event)
        self.s.flush()
        return event

    def get_latest_status(self, run_id: str) -> RunStatusEvent | None:
        return self.s.execute(
            select(RunStatusEvent)
            .where(RunStatusEvent.run_id == run_id)
            .order_by(RunStatusEvent.id.desc())
            .limit(1)
        ).scalar_one_or_none()

    def get_status_history(self, run_id: str) -> list[RunStatusEvent]:
        return list(
            self.s.execute(
                select(RunStatusEvent)
                .where(RunStatusEvent.run_id == run_id)
                .order_by(RunStatusEvent.id.asc())
            ).scalars()
        )


class OptimizationRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def add_run(self, run: OptimizationRun, schedules: list[OptimizationSchedule]) -> None:
        """Insert an optimization run with all scheduled intervals (append-only)."""
        self.s.add(run)
        self.s.flush()
        for s in schedules:
            s.run_id = run.id
        self.s.add_all(schedules)
        self.s.flush()

    def get_run(self, run_id: str) -> OptimizationRun | None:
        return self.s.get(OptimizationRun, run_id)

    def get_schedule(self, run_id: str) -> list[OptimizationSchedule]:
        return list(
            self.s.execute(
                select(OptimizationSchedule)
                .where(OptimizationSchedule.run_id == run_id)
                .order_by(OptimizationSchedule.interval_start_utc)
            ).scalars()
        )

    def list_runs(self, limit: int = 50) -> list[OptimizationRun]:
        return list(
            self.s.execute(
                select(OptimizationRun)
                .order_by(OptimizationRun.created_at.desc())
                .limit(limit)
            ).scalars()
        )


class ShadowRepository:
    """Persistence and audit repository for the Virtual TES Shadow Runtime."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def get_session(self, session_id: str) -> ShadowTESSession | None:
        return self.s.get(ShadowTESSession, session_id)

    def get_latest_session(self) -> ShadowTESSession | None:
        """Return the most recently created or active shadow session."""
        # Prioritize RUNNING or PAUSED session
        active = self.s.execute(
            select(ShadowTESSession)
            .where(ShadowTESSession.status.in_(["RUNNING", "PAUSED"]))
            .order_by(ShadowTESSession.created_at_utc.desc())
            .limit(1)
        ).scalar_one_or_none()
        if active is not None:
            return active
        return self.s.execute(
            select(ShadowTESSession)
            .order_by(ShadowTESSession.created_at_utc.desc())
            .limit(1)
        ).scalar_one_or_none()

    def create_session(self, session_obj: ShadowTESSession) -> ShadowTESSession:
        self.s.add(session_obj)
        self.s.flush()
        return session_obj

    def update_session(self, session_obj: ShadowTESSession) -> None:
        self.s.flush()

    def interval_exists(self, shadow_session_id: str, interval_start_utc: datetime) -> bool:
        start_utc = ensure_utc(interval_start_utc)
        count = self.s.execute(
            select(func.count(ShadowIntervalRecord.id)).where(
                ShadowIntervalRecord.shadow_session_id == shadow_session_id,
                ShadowIntervalRecord.interval_start_utc == start_utc,
            )
        ).scalar_one()
        return count > 0

    def record_interval(self, record: ShadowIntervalRecord) -> ShadowIntervalRecord:
        """Idempotently insert an executed shadow interval.
        
        If an interval with the same (shadow_session_id, interval_start_utc) exists,
        returns the existing record without duplicating.
        """
        record.interval_start_utc = ensure_utc(record.interval_start_utc)
        record.interval_end_utc = ensure_utc(record.interval_end_utc)

        existing = self.s.execute(
            select(ShadowIntervalRecord).where(
                ShadowIntervalRecord.shadow_session_id == record.shadow_session_id,
                ShadowIntervalRecord.interval_start_utc == record.interval_start_utc,
            )
        ).scalar_one_or_none()

        if existing is not None:
            log.info(
                "Shadow interval %s already executed for session %s (no-op)",
                record.interval_start_utc.isoformat(),
                record.shadow_session_id,
            )
            return existing

        self.s.add(record)
        self.s.flush()
        return record

    def get_intervals(
        self,
        shadow_session_id: str,
        limit: int = 500,
        ascending: bool = True,
    ) -> list[ShadowIntervalRecord]:
        stmt = select(ShadowIntervalRecord).where(ShadowIntervalRecord.shadow_session_id == shadow_session_id)
        if ascending:
            stmt = stmt.order_by(ShadowIntervalRecord.interval_start_utc.asc())
        else:
            stmt = stmt.order_by(ShadowIntervalRecord.interval_start_utc.desc())
        if limit:
            stmt = stmt.limit(limit)
        return list(self.s.execute(stmt).scalars())

    def record_audit(self, audit: ShadowEventAudit) -> ShadowEventAudit:
        self.s.add(audit)
        self.s.flush()
        return audit

    def get_audits(self, shadow_session_id: str, limit: int = 100) -> list[ShadowEventAudit]:
        return list(
            self.s.execute(
                select(ShadowEventAudit)
                .where(ShadowEventAudit.shadow_session_id == shadow_session_id)
                .order_by(ShadowEventAudit.timestamp_utc.desc())
                .limit(limit)
            ).scalars()
        )

