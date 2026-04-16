"""
build_db.py
-----------
Converts data/processed/*.csv → soccer_stats.db (SQLite).

Run after build_processed.py:
    python build_db.py

Key optimisation: events.csv (821 MB, 33 cols) is slimmed to the 15 columns
actually used by dashboard.py and scoring.py before writing — reducing the
events table from ~821 MB CSV to ~150–180 MB in the database.

The resulting soccer_stats.db is the deployment artifact committed to git
(via Git LFS if > 100 MB) for Streamlit Community Cloud.
"""

import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
DB_PATH   = ROOT / "soccer_stats.db"

# ── Events: only these 15 columns are used by dashboard.py / scoring.py ───────
EVENTS_KEEP = [
    "game_id",       # WhoScored game ID (join key)
    "season",        # e.g. "2024/25"
    "match_date",    # ISO date string
    "home_team",
    "away_team",
    "team",
    "player",
    "period",        # "FirstHalf" / "SecondHalf"
    "minute",
    "type",          # "Pass", "Tackle", "Shot", …
    "outcome_type",  # "Successful" / "Unsuccessful"
    "x", "y",        # event origin coordinates [0, 100]
    "end_x", "end_y",# event destination coordinates (arrows)
]


def log(msg: str) -> None:
    print(f"[build_db] {msg}", flush=True)


def load_csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        log(f"WARNING: {name} not found — skipping.")
        return pd.DataFrame()
    log(f"Reading {name} …")
    t0 = time.time()
    df = pd.read_csv(path, low_memory=False)
    log(f"  → {len(df):,} rows, {len(df.columns)} cols  ({time.time()-t0:.1f}s)")
    return df


def write_table(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table: str,
    if_exists: str = "replace",
    chunksize: int = 50_000,
) -> None:
    if df.empty:
        log(f"  Skipping {table} — DataFrame is empty.")
        return
    log(f"Writing {table} ({len(df):,} rows) …")
    t0 = time.time()
    df.to_sql(table, conn, if_exists=if_exists, index=False, chunksize=chunksize)
    log(f"  → done  ({time.time()-t0:.1f}s)")


def create_indexes(conn: sqlite3.Connection) -> None:
    log("Creating indexes …")
    stmts = [
        # match_crossref
        "CREATE INDEX IF NOT EXISTS idx_mcr_season   ON match_crossref(season)",
        "CREATE INDEX IF NOT EXISTS idx_mcr_teams    ON match_crossref(home_team, away_team)",
        "CREATE INDEX IF NOT EXISTS idx_mcr_date     ON match_crossref(match_date)",
        # events  (largest table — indexes are critical for scoring.py)
        "CREATE INDEX IF NOT EXISTS idx_ev_game         ON events(game_id)",
        "CREATE INDEX IF NOT EXISTS idx_ev_season_team  ON events(season, team)",
        "CREATE INDEX IF NOT EXISTS idx_ev_type         ON events(type)",
        "CREATE INDEX IF NOT EXISTS idx_ev_player       ON events(player, team, season)",
        # shots
        "CREATE INDEX IF NOT EXISTS idx_sh_season_team  ON shots(season, team)",
        "CREATE INDEX IF NOT EXISTS idx_sh_player       ON shots(player_id)",
        # lineups
        "CREATE INDEX IF NOT EXISTS idx_lu_season_team  ON lineups(season, team)",
        "CREATE INDEX IF NOT EXISTS idx_lu_position     ON lineups(position)",
        "CREATE INDEX IF NOT EXISTS idx_lu_player       ON lineups(player, team, season)",
        # match_summary
        "CREATE INDEX IF NOT EXISTS idx_ms_season       ON match_summary(season)",
        "CREATE INDEX IF NOT EXISTS idx_ms_teams        ON match_summary(home_team, away_team)",
        # player_season
        "CREATE INDEX IF NOT EXISTS idx_ps_season_team  ON player_season(season, primary_team)",
    ]
    for stmt in stmts:
        conn.execute(stmt)
    conn.commit()
    log(f"  → {len(stmts)} indexes created.")


def main() -> None:
    t_start = time.time()
    log(f"Output: {DB_PATH}")

    if DB_PATH.exists():
        log("Removing existing soccer_stats.db …")
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    # WAL mode: safer for concurrent reads, faster writes
    conn.execute("PRAGMA journal_mode=WAL")
    # Larger page size suits wide rows in match_summary
    conn.execute("PRAGMA page_size=4096")

    try:
        # ── match_crossref ────────────────────────────────────────────────────
        write_table(conn, load_csv("match_crossref.csv"), "match_crossref")

        # ── match_summary ─────────────────────────────────────────────────────
        write_table(conn, load_csv("match_summary.csv"), "match_summary")

        # ── player_season ─────────────────────────────────────────────────────
        write_table(conn, load_csv("player_season.csv"), "player_season")

        # ── shots ─────────────────────────────────────────────────────────────
        write_table(conn, load_csv("shots.csv"), "shots")

        # ── lineups ───────────────────────────────────────────────────────────
        write_table(conn, load_csv("lineups.csv"), "lineups")

        # ── events (slimmed to 15 cols) ───────────────────────────────────────
        events_raw = load_csv("events.csv")
        if not events_raw.empty:
            keep = [c for c in EVENTS_KEEP if c in events_raw.columns]
            dropped = set(events_raw.columns) - set(keep)
            if dropped:
                log(f"  Dropping {len(dropped)} unused events columns: {sorted(dropped)}")
            events_slim = events_raw[keep]
            write_table(conn, events_slim, "events")

        # ── indexes ───────────────────────────────────────────────────────────
        create_indexes(conn)

        # ── VACUUM to compact and verify ──────────────────────────────────────
        log("Running VACUUM …")
        conn.execute("VACUUM")
        conn.commit()

    finally:
        conn.close()

    size_mb = DB_PATH.stat().st_size / 1_048_576
    log(f"\n✅  soccer_stats.db  {size_mb:.1f} MB  (built in {time.time()-t_start:.1f}s)")
    if size_mb > 100:
        log("⚠️   File exceeds GitHub's 100 MB limit.")
        log("    Track it with Git LFS before committing:")
        log("    git lfs install")
        log("    git lfs track 'soccer_stats.db'")
        log("    git add .gitattributes soccer_stats.db")


if __name__ == "__main__":
    main()
