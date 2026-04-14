# Soccer Stats App

A data collection and processing pipeline for English Premier League statistics, aggregating data from three independent sources into clean, analysis-ready datasets for heatmaps, shot maps, and predictive modelling.

## Data Sources

| Source | Data collected | Method |
|--------|---------------|--------|
| **Understat** | Match xG, shot coordinates, player season stats | Selenium scraper |
| **ESPN** | Match stats, team lineups, schedules | soccerdata API |
| **WhoScored** | Full match event streams (passes, tackles, pressures, etc.) | soccerdata + Selenium |

## Project Structure

```
Soccer_Stats_App/
├── src/
│   ├── collectors/
│   │   ├── understat_scraper.py    # Incremental match/shot/player scraper
│   │   ├── espn_collector.py       # Incremental schedule, stats, lineup collector
│   │   └── whoscored_collector.py  # Incremental per-match event stream collector
│   ├── sanitize.py                 # Team name canonicalization, coord/date normalization
│   └── utils.py                    # Shared utilities
├── build_processed.py              # Builds analysis-ready datasets from raw data
├── run_collection.py               # Master runner — calls all collectors
├── requirements.txt
└── data/                           # gitignored — generated locally
    ├── raw_understat/
    ├── raw_espn/
    ├── raw_whoscored/
    └── processed/                  # Output of build_processed.py
```

## Setup

**Requirements**: Python 3.11, Google Chrome

```bash
python -m venv soccer_env
source soccer_env/bin/activate      # Windows: soccer_env\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Collect data

```bash
python run_collection.py
```

Runs all three collectors. Only fetches matches not already in the local data files (incremental). WhoScored and Understat will open a browser window — this is intentional to avoid bot detection.

**First run**: expect 1–2 hours for WhoScored (full season event streams). Weekly runs typically take 5–15 minutes.

### 2. Build processed datasets

```bash
python build_processed.py
```

Reads from `data/raw_*/`, normalizes team names and coordinates, links all three source match IDs via a crossref table, and writes six analysis-ready CSVs to `data/processed/`.

Run this after every collection cycle.

## Processed Output Files

| File | Rows | Description |
|------|------|-------------|
| `match_crossref.csv` | ~380 | Links Understat / ESPN / WhoScored match IDs |
| `shots.csv` | ~8k | Shot-level data. Coords in `[0, 100]`. xG per shot |
| `events.csv` | ~500k | Full WhoScored event stream with crossref IDs |
| `match_summary.csv` | ~380 | Per-match xG, goals, forecasts, ESPN stats |
| `lineups.csv` | ~13k | ESPN lineup data with formation positions |
| `player_season.csv` | ~500 | Understat player season stats |

### Coordinate system

All coordinates use the WhoScored `[0, 100]` standard:
- `x = 0` → defending goal line, `x = 100` → attacking goal line
- `y = 0` → top touchline, `y = 100` → bottom touchline
- Both sources share this orientation — Understat values are simply multiplied by 100

## Season Configuration

Update the season variables in both files each season:

**`run_collection.py`**
```python
FBREF_SEASON     = "2526"   # FBref two-year format
ESPN_SEASON      = "2025"   # ESPN end-year
UNDERSTAT_YEAR   = 2025     # Understat start year (int)
WHOSCORED_SEASON = "2025"   # WhoScored start year
```

**`build_processed.py`**
```python
US_SEASON   = 2025
ESPN_SEASON = "2025"
WS_SEASON   = "2025"
```

## Key Design Decisions

- **Incremental collection**: all collectors check existing data before fetching, appending only new matches. Safe to re-run at any time.
- **Canonical team names**: short-form names (e.g. `Man City`, `Wolves`) used across all processed output. Mapping lives in `src/sanitize.py`.
- **Transfer players**: Understat lists multi-team players as `"Bournemouth,Man City"`. The `player_season.csv` adds a `primary_team` column (most recent club).
- **Browser automation**: WhoScored and Understat use Selenium with undetected-chrome (`uc=True`) to bypass bot detection. `headless=False` is default for WhoScored.
