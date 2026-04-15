"""
run_collection.py
-----------------
Master runner for all data collectors.

Loops over every season in SEASONS and runs all three collectors for each.
All collectors are incremental — they skip matches already on disk — so this
script is safe to re-run at any time.  New matches are appended; nothing is
overwritten.

First-time backfill across 5 seasons:
    Understat   ~5-10 min  (fast, JSON scrape)
    ESPN        ~10-20 min (API, rate-limited)
    WhoScored   ~5-10 hrs  (full event stream per match, browser-based)

Weekly update (current season only, ~5-10 new matches):
    Typically 10-30 minutes total.

To collect only the current season, comment out older entries in SEASONS.
"""

from src.collectors.understat_scraper import pull_understat_data
from src.collectors.espn_collector    import pull_espn_data
from src.collectors.whoscored_collector import pull_whoscored_events
from src.utils import ensure_dir

# ---------------------------------------------------------------------------
# Season config
# ---------------------------------------------------------------------------
# Add a new entry here each season — nothing else needs to change.
# Each integer is the Understat start-year (e.g. 2025 = the 2025/26 season).
# ESPN and WhoScored use the same year as a string.
#
# To do a fast weekly update, you can temporarily comment out older seasons.

SEASONS = [
    2023,   # 2023/24
    #2024,   # 2024/25
   #2025,   # 2025/26
    # 2026, # 2026/27  — uncomment when that season starts
]

LEAGUE_ESPN = "ENG-Premier League"
LEAGUE_WS   = "ENG-Premier League"
LEAGUE_US   = "EPL"

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------

ensure_dir("data/raw_understat")
ensure_dir("data/raw_espn")
ensure_dir("data/raw_whoscored")
ensure_dir("logs")

# ---------------------------------------------------------------------------
# Run collectors
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    for us_year in SEASONS:
        espn_season = str(us_year)
        ws_season   = str(us_year)
        label       = f"{us_year - 1}/{str(us_year)[2:]}"

        print(f"\n{'='*55}")
        print(f"Season: {label}  (year={us_year})")
        print(f"{'='*55}")

        # --- Understat ---
        # Pulls match list, player season xG/xA, and shot coordinates.
        # Incremental: only fetches matches not already in the CSV.
        pull_understat_data(
            league=LEAGUE_US,
            season=us_year,
            include_shots=True,
        )

        # --- ESPN ---
        # Pulls schedule, match stats, and team lineups.
        # Incremental: skips games already present in the output files.
        pull_espn_data(season=espn_season, league=LEAGUE_ESPN)

        # --- WhoScored ---
        # Pulls full event stream (passes, shots, tackles, etc.) per match.
        # Incremental: skips matches already in the events CSV.
        # A browser window will open — this is intentional to avoid bot detection.
        pull_whoscored_events(season=ws_season, league=LEAGUE_WS)
