"""
espn_collector.py
-----------------
Pulls schedule, match stats, and team lineups from ESPN for the selected
league and season.

Incremental behaviour:
    - Schedule: always re-fetched and overwritten (lightweight; needed to
      discover the game-string → game_id mapping for new matches).
    - Match stats / Lineups: check which games are already in the existing
      CSV, then fetch and append only the new ones.

Stale cache handling:
    ESPN summary JSON files are cached by soccerdata before a match is
    played (no lineup data yet). _bust_stale_lineup_cache() deletes those
    stubs so completed matches are re-fetched from the live API.
"""

import json
import soccerdata as sd
import pandas as pd
from pathlib import Path
from src.utils import ensure_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bust_stale_lineup_cache(data_dir: Path) -> int:
    """
    Delete cached ESPN Summary JSON files where roster entries are missing
    the actual 'roster' key — meaning they were fetched before the match
    was played and contain no lineup data.

    Returns the number of files cleared.
    """
    cleared = 0
    for path in data_dir.glob("Summary_*.json"):
        try:
            data = json.loads(path.read_text())
            rosters = data.get("rosters", [])
            if rosters and all("roster" not in r for r in rosters):
                path.unlink()
                cleared += 1
        except Exception:
            pass
    return cleared


def _load_collected_games(filepath: Path) -> set:
    """
    Returns the set of 'game' string IDs already present in a CSV.
    Returns an empty set if the file doesn't exist or has no 'game' column
    (e.g. an older file saved without the index).
    """
    if not filepath.exists():
        return set()
    try:
        df = pd.read_csv(filepath, usecols=["game"])
        ids = set(df["game"].dropna().unique())
        print(f"  {len(ids)} games already in {filepath.name}")
        return ids
    except Exception as exc:
        print(f"  ⚠ Could not read {filepath.name} for deduplication: {exc}")
        print(f"  Will re-fetch all matches this run.")
        return set()


def _append_to_csv(new_df: pd.DataFrame, filepath: Path) -> None:
    """Append new rows to a CSV, writing the header only when creating the file."""
    write_header = not filepath.exists()
    new_df.to_csv(filepath, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

def pull_espn_data(season="2026", league="ENG-Premier League"):
    """
    Incrementally pulls schedule, match stats, and lineups from ESPN.
    Only fetches data for matches not already present in the output CSVs.

    Parameters
    ----------
    season : str   ESPN season string (e.g. "2025" for 2024-25).
    league : str   ESPN league string (e.g. "ENG-Premier League").
    """
    print(f"--- Initializing ESPN for {league} ({season}) ---")
    espn = sd.ESPN(leagues=[league], seasons=[season])
    ensure_dir("data/raw_espn")

    try:
        # --- Schedule: always re-fetch ---
        # Lightweight single request; provides the game-string → game_id
        # integer mapping needed to call read_matchsheet / read_lineup
        # incrementally.
        print("\nFetching team schedule...")
        schedule_df = espn.read_schedule().reset_index()
        schedule_filename = f"data/raw_espn/{league}_{season}_team_schedule.csv"
        schedule_df.to_csv(schedule_filename, index=False)
        print(f"  Saved → {schedule_filename}")

        # Filter to completed matches (kick-off date in the past)
        today = pd.Timestamp.now(tz="UTC")
        completed = schedule_df[pd.to_datetime(schedule_df["date"], utc=True) <= today].copy()
        game_to_id = dict(zip(completed["game"], completed["game_id"].astype(int)))
        print(f"  {len(game_to_id)} completed matches found")

        if not game_to_id:
            print("  No completed matches yet — nothing to fetch.")
            return

        # --- Match stats: incremental ---
        print("\nFetching match stats...")
        ms_path = Path(f"data/raw_espn/{league}_{season}_match_stats.csv")
        existing_ms = _load_collected_games(ms_path)
        new_ms = {g: gid for g, gid in game_to_id.items() if g not in existing_ms}

        if new_ms:
            print(f"  Fetching {len(new_ms)} new match(es)...")
            new_ms_df = espn.read_matchsheet(match_id=list(new_ms.values()))
            _append_to_csv(new_ms_df.reset_index(), ms_path)
            print(f"  Appended {len(new_ms)} match(es) → {ms_path.name}")
        else:
            print("  ✓ Match stats already up to date")

        # --- Lineups: incremental + stale cache busting ---
        print("\nFetching team lineups...")
        cleared = _bust_stale_lineup_cache(espn.data_dir)
        if cleared:
            print(f"  Cleared {cleared} stale pre-match cache file(s)")

        lu_path = Path(f"data/raw_espn/{league}_{season}_team_lineups.csv")
        existing_lu = _load_collected_games(lu_path)
        new_lu = {g: gid for g, gid in game_to_id.items() if g not in existing_lu}

        if new_lu:
            print(f"  Fetching {len(new_lu)} new match(es)...")
            new_lu_df = espn.read_lineup(match_id=list(new_lu.values()))
            _append_to_csv(new_lu_df.reset_index(), lu_path)
            print(f"  Appended {len(new_lu)} match(es) → {lu_path.name}")
        else:
            print("  ✓ Lineups already up to date")

    except Exception as e:
        import traceback
        print(f"Could not fetch ESPN data: {e}")
        traceback.print_exc()
