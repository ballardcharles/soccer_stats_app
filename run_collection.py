"""
run_collection.py
-----------------
Master runner for all data collectors.
Run this script weekly to keep all data up to date.

Data collected:
    Understat   — match xG, shot coordinates, player season xG/xA
    ESPN        — schedule, match stats, lineups
    WhoScored   — full match event streams (passes, tackles, pressures, etc.)

Season format note:
    FBref uses "2526" (string, two-year format, e.g. 2025-26)
    ESPN / WhoScored use "2025" (string, end-year or start-year)
    Understat uses 2025 (int, season start year)
"""

from src.collectors.understat_scraper import pull_understat_data
from src.collectors.espn_collector import pull_espn_data
from src.collectors.whoscored_collector import pull_whoscored_events
from src.utils import ensure_dir

# ---------------------------------------------------------------------------
# Season config — update these each season
# ---------------------------------------------------------------------------

ESPN_SEASON    = "2025"   # ESPN end-year format
UNDERSTAT_YEAR = 2025     # Understat start year (int)
WHOSCORED_SEASON = "2025" # WhoScored start year (string)

LEAGUE_ESPN    = "ENG-Premier League"
LEAGUE_WS      = "ENG-Premier League"
LEAGUE_US      = "EPL"

# ---------------------------------------------------------------------------
# Ensure all output directories exist
# ---------------------------------------------------------------------------

ensure_dir("data/raw_understat")
ensure_dir("data/raw_espn")
ensure_dir("data/raw_whoscored")
ensure_dir("logs")

# ---------------------------------------------------------------------------
# Run collectors
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Understat ---
    # Pulls: match list + player season xG/xA (one page load),
    #        shot coordinates for each new completed match (incremental)
    # Full season shot scrape: ~30-40 min first run, ~5 min weekly thereafter
    pull_understat_data(
        league=LEAGUE_US,
        season=UNDERSTAT_YEAR,
        include_shots=True,
    )

    # --- ESPN ---
    # Pulls: schedule, match stats, team lineups
    pull_espn_data(season=ESPN_SEASON, league=LEAGUE_ESPN)

    # --- WhoScored ---
    # Pulls: full event stream for each match (passes, shots, tackles,
    #        dribbles, pressures, clearances, carries)
    # Note: headless=False is set inside the collector — a browser window
    #       will open. This is intentional to avoid bot detection.
    # First run is slow (~1-2 hrs for full season). Weekly runs are fast.
    pull_whoscored_events(season=WHOSCORED_SEASON, league=LEAGUE_WS)