"""
espn_collector.py
-----------------
Pulls schedule, match stats, and team lineups from ESPN for the selected
league and season.

Incremental behaviour:
    - Schedule: always re-fetched and overwritten (lightweight; needed to
      discover the game-string → game_id mapping for new matches).
    - Match stats / Lineups: check which game_ids are already in the existing
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


def _load_collected_game_ids(filepath: Path) -> set:
    """
    Returns the set of integer game_ids already present in a CSV.
    Tries 'game_id' column first, then 'game' (old multi-index format).
    Returns an empty set if the file doesn't exist or cannot be parsed.
    """
    if not filepath.exists():
        return set()
    for col in ("game_id", "game"):
        try:
            df = pd.read_csv(filepath, usecols=[col])
            vals = df[col].dropna()
            # If the column holds integer game_ids, return them as ints.
            # If it holds game strings (old format), fall through to next col.
            if col == "game_id":
                ids = set(vals.astype(int).unique())
                print(f"  {len(ids)} games already in {filepath.name}")
                return ids
            else:
                # Old format: 'game' column holds strings like
                # "2024-08-16 Manchester United-Fulham".
                # We can't directly compare those to integer game_ids, so
                # return them as-is and let the caller handle the key type.
                ids = set(vals.unique())
                print(f"  {len(ids)} games already in {filepath.name} (legacy 'game' col)")
                return ids
        except Exception:
            pass
    print(f"  ⚠ Could not read {filepath.name} for deduplication — will re-fetch.")
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
    season : str   ESPN season string (end-year convention: "2026" = 2025/26).
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
        # Build both lookup directions
        game_to_id   = dict(zip(completed["game"], completed["game_id"].astype(int)))
        id_to_game   = {v: k for k, v in game_to_id.items()}
        print(f"  {len(game_to_id)} completed matches found")

        if not game_to_id:
            print("  No completed matches yet — nothing to fetch.")
            return

        # --- Match stats: incremental ---
        print("\nFetching match stats...")
        ms_path = Path(f"data/raw_espn/{league}_{season}_match_stats.csv")
        existing = _load_collected_game_ids(ms_path)

        # existing may hold int game_ids (new format) or game strings (old format)
        if existing and isinstance(next(iter(existing)), int):
            new_ms = {g: gid for g, gid in game_to_id.items() if gid not in existing}
        else:
            new_ms = {g: gid for g, gid in game_to_id.items() if g not in existing}

        if new_ms:
            print(f"  Fetching {len(new_ms)} new match(es)...")
            new_ms_df = espn.read_matchsheet(match_id=list(new_ms.values()))
            ms_out = new_ms_df.reset_index()
            # Explicitly add game_id so future runs can deduplicate reliably.
            if "game_id" not in ms_out.columns:
                ids_order = list(new_ms.values())
                repeated   = [gid for gid in ids_order for _ in range(2)]
                if len(ms_out) == len(repeated):
                    ms_out.insert(0, "game_id", repeated)
            _append_to_csv(ms_out, ms_path)
            print(f"  Appended {len(new_ms)} match(es) → {ms_path.name}")
        else:
            print("  ✓ Match stats already up to date")

        # --- Lineups: incremental + stale cache busting ---
        print("\nFetching team lineups...")
        cleared = _bust_stale_lineup_cache(espn.data_dir)
        if cleared:
            print(f"  Cleared {cleared} stale pre-match cache file(s)")

        lu_path = Path(f"data/raw_espn/{league}_{season}_team_lineups.csv")
        existing_lu = _load_collected_game_ids(lu_path)

        if existing_lu and isinstance(next(iter(existing_lu)), int):
            new_lu = {g: gid for g, gid in game_to_id.items() if gid not in existing_lu}
        else:
            new_lu = {g: gid for g, gid in game_to_id.items() if g not in existing_lu}

        if new_lu:
            print(f"  Fetching {len(new_lu)} new match(es)...")
            new_lu_df = espn.read_lineup(match_id=list(new_lu.values()))
            lu_out = new_lu_df.reset_index()
            # Explicitly add game_id for future dedup reliability.
            if "game_id" not in lu_out.columns:
                ids_order = list(new_lu.values())
                # Lineup has multiple rows per game (one per player ≈ 22+).
                # We can't know exact row count per game up front, so only
                # stamp game_id if the 'game' string column is present (old
                # multi-index format) or skip — build_processed handles it.
                if "game" in lu_out.columns:
                    lu_out.insert(0, "game_id",
                                  lu_out["game"].map({id_to_game[gid]: gid
                                                      for gid in ids_order
                                                      if gid in id_to_game}))
            _append_to_csv(lu_out, lu_path)
            print(f"  Appended {len(new_lu)} match(es) → {lu_path.name}")
        else:
            print("  ✓ Lineups already up to date")

    except Exception as e:
        import traceback
        print(f"Could not fetch ESPN data: {e}")
        traceback.print_exc()
