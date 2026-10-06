# Virtual TES Gen0 — Railway 24/7 Cloud Deployment Guide

================================================================================
PHASE 5.9 — PRODUCTION RAILWAY ARCHITECTURE & RUNBOOK
================================================================================

## 1. Target Architecture Overview

The production Virtual TES Gen0 plant is deployed on Railway as an autonomous 3-service architecture:

```
                            [ Railway Project ]
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
[ WEB SERVICE ]             [ WORKER SERVICE ]             [ POSTGRES SERVICE ]
FastAPI + Jinja/HTMX        Autonomous 24/7 Engine         Managed PostgreSQL DB
Engineering Dashboard       (No public HTTP port)          Persistent Source of Truth
HTTPS: public domain        Advisory Lock Leader           DATABASE_URL
Auth: ADMIN & VIEWER        Litgrid & Elering Poller       18 Tables + Constraints
Read-only & Mutation APIs   Replay Recovery Loop           Append-Only Integrity
```

### Critical Digital Twin Constraints
- **Zero Physical Hardware Control:** Physical Gen0 heating elements and PLC are **NOT CONNECTED**. All thermal and electrical actuators operate inside the software simulation model.
- **Laptop Independence:** The system runs continuously 24/7 on Railway cloud infrastructure without requiring an open browser, terminal, or local computer.
- **Canonical State Persistence:** State of Charge (SOC), stored energy, sand temperature, and interval history reside in PostgreSQL. Service restarts or redeployments resume from the database without re-initializing state.

---

## 2. Step-by-Step Railway Deployment Instructions

### Step 1: Create Railway Project
1. Log in to [Railway](https://railway.com).
2. Click **New Project** &rarr; select **Deploy from GitHub repo**.
3. Choose the `virtual-tes` repository.

### Step 2: Add PostgreSQL Database
1. Within your project canvas, click **Create** &rarr; **Database** &rarr; **Add PostgreSQL**.
2. Railway creates a managed PostgreSQL instance and automatically injects the `DATABASE_URL` variable.

### Step 3: Configure WEB Service
1. Select the service created from your GitHub repo and rename it to `web`.
2. Navigate to **Settings** &rarr; **Deploy**:
   - **Custom Build Command:** *(leave empty or default — handled by Dockerfile)*
   - **Custom Start Command:**
     ```bash
     uvicorn app.web.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Healthcheck Path:** `/health`
   - **Healthcheck Timeout:** `100` seconds
3. Navigate to **Settings** &rarr; **Networking**:
   - Click **Generate Domain** to assign a public HTTPS address (e.g., `https://virtual-tes-production.up.railway.app`).

### Step 4: Configure WORKER Service
1. In the project canvas, click **Create** &rarr; **GitHub Repo** &rarr; select the same `virtual-tes` repo again.
2. Rename this second service to `worker`.
3. Navigate to **Settings** &rarr; **Deploy**:
   - **Custom Start Command:**
     ```bash
     python -m app.worker
     ```
   - **Healthcheck Path:** *(leave empty — worker runs as a dedicated headless process)*
4. Ensure **Public Networking** is **disabled** for the worker service.

### Step 5: Configure Environment Variables
Set the following Railway Variables on **both** the `web` and `worker` services (or use Railway Shared Variables):

| Variable Name | Value | Purpose |
| :--- | :--- | :--- |
| `APP_ENV` | `production` | Enables strict production validations & auth |
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Connection to persistent PostgreSQL |
| `RUN_IN_PROCESS_WORKER` | `false` | Prevents WEB service from duplicating worker loop |
| `MARKET_PROVIDER` | `LITGRID` | Primary Lithuanian day-ahead data source |
| `LITGRID_API_URL` | `https://openapi.litgrid.eu/v1/kategorijos/elektros-energijos-kainos/801` | Official Litgrid API |
| `ELERING_API_URL` | `https://dashboard.elering.ee/api/nps/price` | Official Elering validation API |
| `APP_SECRET_KEY` | *(32+ char random hex string)* | Cookie HMAC-SHA256 signature key |
| `AUTH_REQUIRED` | `true` | Enforces authentication on all dashboard routes |
| `DASHBOARD_USERNAME` | `admin` | Administrator username |
| `DASHBOARD_PASSWORD_HASH` | `pbkdf2_sha256$...` | PBKDF2 hash of admin password |
| `VIEWER_USERNAME` | `viewer` | Read-only viewer username |
| `VIEWER_PASSWORD_HASH` | `pbkdf2_sha256$...` | PBKDF2 hash of viewer password |
| `TES_TIMEZONE` | `Europe/Vilnius` | Local Lithuanian market timezone |
| `TES_BIDDING_ZONE` | `LT` | Nord Pool Lithuania price area |

> **Password Hash Generation:**
> Run locally using uv:
> ```bash
> uv run python -m app.auth.cli hash "YourSecureAdminPassword"
> uv run python -m app.auth.cli hash "YourGuestViewerPassword"
> ```
> Copy the resulting `pbkdf2_sha256$...` string into the respective variable.

### Step 6: Run Database Migrations
In the Railway web terminal for either service (or via Railway CLI):
```bash
python -m app.database.migrate
```
This idempotently verifies and initializes all 18 tables without dropping or altering existing rows.

---

## 3. Single-Active Worker & Distributed Lock Guarantees

To ensure that two worker processes never execute the live shadow TES concurrently:
1. The worker acquires a PostgreSQL session-level advisory lock:
   ```sql
   SELECT pg_try_advisory_lock(847293);
   ```
2. If acquired (`True`), the instance becomes the **ACTIVE LEADER**:
   - Executes interval boundaries.
   - Runs continuous physical state evolution.
   - Triggers receding-horizon LP optimization.
   - Writes `WorkerHeartbeat` with status `RUNNING`.
3. If held by another instance (`False`), the secondary worker transitions to **STANDBY**:
   - Writes `WorkerHeartbeat` with status `STANDBY`.
   - Never executes intervals or mutates TES state.
4. When the leader terminates (SIGTERM or container teardown), PostgreSQL automatically releases the advisory lock within the closed session, allowing standby or replacement containers to take over cleanly.

---

## 4. Restart Recovery & Data Integrity Runbook

When Railway performs a rolling restart or deployment:
1. **State Preservation:** The latest SOC, bulk sand temperature, stored energy, and schedule version remain in `shadow_tes_sessions`.
2. **Replay Catch-up:** Upon worker restart:
   ```python
   shadow_service.catch_up_missed_intervals(session, active_session)
   ```
   The worker queries the timestamp of the last executed interval and replays any completed 15-minute market intervals missed during downtime.
3. **Idempotence Enforcement:** The database constraint:
   ```sql
   UNIQUE(shadow_session_id, interval_start_utc)
   ```
   strictly blocks duplicate interval execution.
4. **Heartbeat Freshness:**
   - Worker writes a heartbeat every 60 seconds.
   - If heartbeat is older than 180 seconds (3 minutes), the UI displays `WORKER OFFLINE` and suspends the displayed active status.

---

## 5. PostgreSQL Backup & Restore Strategy

### Creating a Snapshot Backup (pg_dump)
Run this command from your local machine or an automated backup job using Railway's database connection credentials:
```bash
pg_dump "postgresql://postgres:PASSWORD@HOST:PORT/railway" \
  --format=custom \
  --no-owner \
  --no-privileges \
  --file="virtual_tes_backup_$(date +%Y%m%d_%H%M%S).dump"
```

### Restoring from Backup (pg_restore)
In the event of database disaster recovery:
```bash
pg_restore \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  --dbname="postgresql://postgres:PASSWORD@HOST:PORT/railway" \
  "virtual_tes_backup_YYYYMMDD_HHMMSS.dump"
```

### Pre-Migration Safeguard
Before applying future major schema alterations, take a point-in-time snapshot with `pg_dump`.

---

## 6. Access Control & Role Permissions

| Action / Page | ADMIN Role | VIEWER Role | Unauthenticated |
| :--- | :---: | :---: | :---: |
| Overview / Market / Physical / HX / Vessel / Dispatch | ✅ | ✅ | ❌ (Redirect to `/login`) |
| Economic Comparison / Sizing / Sensitivity / Assumptions | ✅ | ✅ | ❌ (Redirect to `/login`) |
| GET `/api/shadow/status`, `/shadow/history`, `/shadow/live-card` | ✅ | ✅ | ❌ (HTTP 401) |
| GET `/health` | ✅ | ✅ | ✅ (Public healthcheck) |
| POST `/api/shadow/start` | ✅ | ❌ (HTTP 403 Forbidden) | ❌ (HTTP 401) |
| POST `/api/shadow/pause` / `/shadow/resume` / `/shadow/stop` | ✅ | ❌ (HTTP 403 Forbidden) | ❌ (HTTP 401) |
| POST `/api/shadow/reset` | ✅ | ❌ (HTTP 403 Forbidden) | ❌ (HTTP 401) |
| POST `/api/shadow/reoptimize` | ✅ | ❌ (HTTP 403 Forbidden) | ❌ (HTTP 401) |
| POST `/api/scenarios` (Save Scenario) | ✅ | ❌ (HTTP 403 Forbidden) | ❌ (HTTP 401) |
