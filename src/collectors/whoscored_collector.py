"""
whoscored_collector.py
----------------------
Incrementally scrapes full match event streams from WhoScored.com
for the selected league and season using the soccerdata library.

How incremental mode works:
    1. Fetches the season schedule from WhoScored (fast, one browser session).
    2. Filters to completed matches only (kick-off date in the past).
    3. Reads the existing events CSV to find match IDs already collected.
    4. Fetches events only for new matches, appending to the CSV after
       each match so progress is saved even if the run is interrupted.

Bot detection notes:
    - headless=False is the default — WhoScored's Incapsula protection is
      much easier to pass with a visible browser window.
    - Delays between matches are handled internally by soccerdata.
    - If you see 403 errors: close all Chrome windows, wait 10 minutes,
      then re-run.
"""

import soccerdata as sd
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.utils import ensure_dir

MAX_CONSECUTIVE_FAILURES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_collected_ids(events_path: Path) -> set:
    """Returns the set of game_ids already present in the events CSV."""
    if not events_path.exists():
        return set()
    try:
        df = pd.read_csv(events_path, usecols=["game_id"])
        ids = set(df["game_id"].astype(str).unique())
        print(f"  {len(ids)} matches already collected in {events_path.name}")
        return ids
    except Exception as exc:
        print(f"  ⚠ Could not read existing events file: {exc}. Starting fresh.")
        return set()


def _append_events(events_df: pd.DataFrame, events_path: Path) -> None:
    """Appends new event rows to the CSV, deduplicating by game_id."""
    if events_path.exists() and "game_id" in events_df.columns:
        try:
            existing_ids = set(
                pd.read_csv(events_path, usecols=["game_id"])["game_id"].astype(str)
            )
            events_df = events_df[~events_df["game_id"].astype(str).isin(existing_ids)]
            if events_df.empty:
                print("    (already in events file — skipping duplicate write)")
                return
        except Exception as exc:
            print(f"    ⚠ Could not check for duplicates: {exc}. Writing anyway.")
    write_header = not events_path.exists()
    events_df.to_csv(events_path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

def pull_whoscored_events(league="ENG-Premier League", season="2025"):
    """
    Incrementally fetch WhoScored event stream data for the selected
    league and season. Only collects matches not already in the output CSV.

    Parameters
    ----------
    league : str   WhoScored league name (e.g. "ENG-Premier League").
    season : str   Season start year as a string (e.g. "2025" for 2024-25).
    """
    print(f"\n{'='*50}")
    print(f"WhoScored collection started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"League: {league}  |  Season: {season}")
    print(f"{'='*50}")

    ensure_dir("data/raw_whoscored")

    events_path = Path(f"data/raw_whoscored/{league}_{season}_events.csv")
    already_collected = _load_collected_ids(events_path)

    # --- Step 1: fetch the season schedule ---
    print("\n  Initialising WhoScored scraper...")
    try:
        ws = sd.WhoScored(leagues=[league], seasons=[season], headless=False)
        print("  Fetching season schedule...")
        schedule = ws.read_schedule()
    except Exception as exc:
        print(f"  Could not initialise WhoScored or fetch schedule: {exc}")
        print("  TIP: If you see a 403, close all Chrome windows and wait 10 minutes.")
        return

    if schedule.empty:
        print("  No matches found for this season.")
        return

    # Flatten MultiIndex (league, season, game-string) into columns.
    # Save as CSV so build_processed.py can build the match crossref
    # without needing to re-open a browser session.
    flat = schedule.reset_index()
    schedule_path = Path(f"data/raw_whoscored/{league}_{season}_schedule.csv")
    flat.to_csv(schedule_path, index=False)
    print(f"  Saved schedule → {schedule_path.name}")

    # --- Step 2: filter to completed matches not yet collected ---
    # status == 6 means the match has been played (WhoScored convention).

    if "status" in flat.columns:
        completed = flat[flat["status"] == 6].copy()
    else:
        # Fallback: filter by date if status column is unavailable
        today = pd.Timestamp.now(tz="UTC")
        dates = pd.to_datetime(flat.get("date", pd.Series(dtype="object")), errors="coerce", utc=True)
        completed = flat[dates.notna() & (dates <= today)].copy()

    new_matches = completed[~completed["game_id"].astype(str).isin(already_collected)]

    if new_matches.empty:
        print("\n  ✓ WhoScored events already up to date — nothing to fetch.")
        return

    print(f"\n  {len(new_matches)} new match(es) to fetch")
    if already_collected:
        print(f"  (skipping {len(already_collected)} already collected)")

    # --- Step 3: fetch events match by match ---
    total_events = 0
    consecutive_failures = 0

    for i, (_, row) in enumerate(new_matches.iterrows(), start=1):
        match_id = int(row["game_id"])
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        label = f"{home} vs {away}"

        print(f"\n  [{i}/{len(new_matches)}] {label} (id={match_id})")

        try:
            events_df = ws.read_events(match_id=match_id, on_error="skip")

            if events_df is None or (isinstance(events_df, pd.DataFrame) and events_df.empty):
                print("    ⚠ No events returned — match may not be available yet.")
                consecutive_failures += 1
            else:
                # Ensure game_id is present as a column for future deduplication
                if "game_id" not in events_df.columns:
                    events_df.insert(0, "game_id", match_id)
                _append_events(events_df, events_path)
                total_events += len(events_df)
                consecutive_failures = 0
                print(f"    ✓ {len(events_df)} events saved")

        except Exception as exc:
            print(f"    ⚠ Failed: {exc}")
            consecutive_failures += 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\n  {MAX_CONSECUTIVE_FAILURES} consecutive failures — stopping run early.")
            print("  TIP: Close Chrome, wait 10-15 minutes, then re-run.")
            break

    print(f"\n  Appended {total_events} new event row(s) → {events_path}")
    print(f"  Run complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
