# Virtual TES Gen0 — Energy Management System

Virtual Gen0 software platform for a future sand-based Thermal Energy Storage (TES) plant.

This is the virtual simulation and market layer prior to connecting real PLC hardware.

---

## Operating Principle

```
ELECTRICITY GRID
        |
        v
ELECTRIC HEATERS
        |
        v
THERMAL ENERGY STORAGE (TES)
        |
        v
HEAT EXCHANGER
        |
        v
INDUSTRIAL HEAT PROCESS (e.g. Wood Drying Chamber)
```

* **Process Heat Priority**: The heat exchanger delivers whatever thermal power the process demands (e.g. 1.5 kW), regardless of electricity prices.
* **Economic Arbitrage**: Electricity price determines *when* the electric heaters run to charge the thermal battery.
* **Simultaneous Operation**: The TES can charge from the heaters while discharging to the process at the same time.
* **Grid Limit**: Total site load (TES heaters + other site electrical loads) must never exceed the grid connection limit (default: 12 kW).

---

## Quickstart & Setup

### 1. Environment Requirements
- Python 3.12+
- `uv` (recommended) or standard `pip` / `venv`

### 2. Setup Virtual Environment & Dependencies
```powershell
cd C:\Users\kaim\.gemini\antigravity\scratch\virtual-tes

# Create virtual environment and install dependencies
$env:PATH = "$HOME\.local\bin;" + $env:PATH
uv venv
uv pip install -e .
uv pip install pytest
```

### 3. Initialize the Database
Initializes SQLite database tables and installs immutable triggers preventing any `UPDATE` or `DELETE` on historical tables:
```powershell
uv run python init_db.py
```

### 4. Run Automated Tests
```powershell
uv run pytest
```

### 5. Run 24-Hour Simulation Demo
Runs a 24-hour simulation with 15-minute market resolution and outputs the full interval-by-interval table and summary:
```powershell
uv run python simulate_day.py --date 2026-10-06 --strategy cheapest_n
```

Optional arguments:
- `--strategy [cheapest_n|heat_following]`
- `--n-cheapest [N]` (e.g. `--n-cheapest 16`)
- `--save` (saves the run, prices and profiles to SQLite)

---

## Technical Specifications (Gen0 Defaults)

- **TES Total Thermal Capacity**: 15.0 kWh
- **Usable SOC Range**: 10.0% – 100.0% (1.5 kWh – 15.0 kWh; 13.5 kWh usable)
- **Max Heater Power**: 9.0 kW electrical
- **Max Discharge Power**: 3.0 kW thermal
- **Charge Efficiency**: 95%
- **Discharge Efficiency**: 90%
- **Round-Trip Efficiency**: 85.5%
- **Standing Losses**: 2.0% per 24 hours of stored energy ($r = (1 - 0.02)^{\Delta t / 24}$)
- **Process Heat Demand**: 1.5 kW constant (or scheduled blocks)
- **Grid Connection Limit**: 12.0 kW electrical
- **Other Site Electrical Loads**: 2.0 kW electrical
- **Time Resolution**: 15 minutes (96 intervals / standard day)
