"""
build_db.py
-----------
Converts data/processed/*.csv → soccer_stats.db (SQLite).

Run after build_processed.py:
    python build_db.py

Key optimisation: events.csv (821 MB, 33 cols) is loaded with qualifiers
temporarily included, then:
  • Three boolean set-piece columns are extracted from qualifiers before it is
    dropped: is_corner, is_freekick, is_direct_fk
  • FormationSet events are parsed into a separate lightweight `formations` table
  • The final events table keeps 18 columns (~150–180 MB in the database)

WhoScored TeamFormation codes → formation strings are decoded from the
qualifiers of FormationSet events.

The resulting soccer_stats.db is the deployment artifact committed to git
(via Git LFS if > 100 MB) for Streamlit Community Cloud.
"""

import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
DB_PATH   = ROOT / "soccer_stats.db"

# ── Events: columns kept in the final DB events table ─────────────────────────
# (qualifiers is loaded temporarily and then dropped after extraction)
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
    "end_x", "end_y",# event destination coordinates
    # ── Derived from qualifiers (extracted before qualifiers is dropped) ──────
    "is_corner",     # 1 if Pass with CornerTaken qualifier
    "is_freekick",   # 1 if Pass with FreekickTaken qualifier
    "is_direct_fk",  # 1 if Pass/Shot with DirectFreekick qualifier
]

# ── WhoScored TeamFormation code → human-readable string ──────────────────────
# Values reverse-engineered from WhoScored's internal formation API.
WHOSCORED_FORMATIONS: dict[int, str] = {
    2:  "4-4-2",
    3:  "4-3-3",
    4:  "4-2-3-1",
    5:  "3-4-3",
    6:  "5-3-2",
    7:  "4-1-4-1",
    8:  "4-3-3",      # WhoScored uses codes 3 and 8 for 4-3-3 variants
    9:  "3-5-2",
    10: "4-3-1-2",
    11: "4-4-1-1",
    12: "3-4-1-2",
    13: "4-1-3-2",
    15: "5-4-1",
    16: "4-2-2-2",
    17: "4-2-3-1",    # variant of 4-2-3-1
    18: "4-1-2-1-2",
    22: "5-2-3",
    23: "5-3-2",
    24: "3-5-2",
}

# Regex to pull the TeamFormation value from the qualifiers string
_FORM_CODE_RE = re.compile(r"TeamFormation.*?'value':\s*'(\d+)'")


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


def _parse_form_code(qual_str: str) -> int | None:
    """Extract TeamFormation numeric code from a qualifiers string."""
    m = _FORM_CODE_RE.search(str(qual_str))
    return int(m.group(1)) if m else None


def extract_formations(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse FormationSet events from the raw events DataFrame and return a
    lightweight `formations` table:
        game_id, season, match_date, home_team, away_team, team,
        formation_code, formation
    """
    fs = events_df[events_df["type"] == "FormationSet"].copy()
    if fs.empty:
        log("  No FormationSet events found — formations table will be empty.")
        return pd.DataFrame()

    keep_cols = [c for c in
                 ["game_id", "season", "match_date", "home_team", "away_team",
                  "team", "qualifiers"]
                 if c in fs.columns]
    fs = fs[keep_cols].copy()
    fs["formation_code"] = fs["qualifiers"].apply(_parse_form_code)
    fs["formation"] = fs["formation_code"].map(WHOSCORED_FORMATIONS)
    fs = fs.drop(columns=["qualifiers"])

    # Keep only rows where we could decode the formation
    fs = fs.dropna(subset=["formation"])
    log(f"  → {len(fs):,} formation records decoded "
        f"({fs['formation'].value_counts().head(5).to_dict()})")
    return fs.reset_index(drop=True)


def extract_set_piece_flags(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three lightweight boolean (int8) columns derived from qualifiers:
        is_corner    — Pass taken from a corner (CornerTaken qualifier)
        is_freekick  — Pass taken from a free kick (FreekickTaken qualifier)
        is_direct_fk — Direct free kick attempt (DirectFreekick qualifier)
    """
    q = events_df["qualifiers"].fillna("")
    events_df["is_corner"]    = q.str.contains("CornerTaken",    na=False).astype("int8")
    events_df["is_freekick"]  = q.str.contains("FreekickTaken",  na=False).astype("int8")
    events_df["is_direct_fk"] = q.str.contains("DirectFreekick", na=False).astype("int8")

    n_corners = int(events_df["is_corner"].sum())
    n_fk      = int(events_df["is_freekick"].sum())
    n_dfk     = int(events_df["is_direct_fk"].sum())
    log(f"  → set-piece flags: {n_corners:,} corners · {n_fk:,} free kicks · {n_dfk:,} direct FKs")
    return events_df


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
        "CREATE INDEX IF NOT EXISTS idx_ev_corner       ON events(is_corner)",
        "CREATE INDEX IF NOT EXISTS idx_ev_freekick     ON events(is_freekick)",
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
        # formations
        "CREATE INDEX IF NOT EXISTS idx_fm_season_team  ON formations(season, team)",
        "CREATE INDEX IF NOT EXISTS idx_fm_game         ON formations(game_id)",
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

        # ── events ────────────────────────────────────────────────────────────
        # Load with qualifiers so we can extract formations + set-piece flags,
        # then drop qualifiers before writing to keep the DB compact.
        events_path = PROCESSED / "events.csv"
        if events_path.exists():
            log("Reading events.csv (with qualifiers for extraction)…")
            t0 = time.time()
            # Load EVENTS_KEEP cols + qualifiers (needed for extraction)
            base_cols = set(EVENTS_KEEP) - {"is_corner", "is_freekick", "is_direct_fk"}
            events_raw = pd.read_csv(
                events_path,
                usecols=lambda c: c in base_cols or c == "qualifiers",
                low_memory=False,
            )
            log(f"  → {len(events_raw):,} rows, {len(events_raw.columns)} cols  ({time.time()-t0:.1f}s)")

            # ── Extract formations table (must happen before qualifiers drop) ─
            log("Extracting formations from FormationSet events…")
            formations_df = extract_formations(events_raw)
            write_table(conn, formations_df, "formations")

            # ── Extract set-piece boolean flags ───────────────────────────────
            log("Extracting set-piece flags from qualifiers…")
            events_raw = extract_set_piece_flags(events_raw)

            # ── Drop qualifiers, keep only EVENTS_KEEP columns ────────────────
            events_slim = events_raw[[c for c in EVENTS_KEEP if c in events_raw.columns]]
            log(f"  → slimmed to {len(events_slim.columns)} cols (qualifiers dropped)")
            write_table(conn, events_slim, "events")
        else:
            log("WARNING: events.csv not found — skipping events + formations tables.")

        # ── indexes ───────────────────────────────────────────────────────────
        create_indexes(conn)

        # ── Switch to DELETE journal mode before VACUUM ───────────────────────
        # WAL mode leaves .db-wal/.db-shm sidecar files; switching to DELETE
        # ensures the committed DB is a single clean file with no sidecars.
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.commit()

        # ── VACUUM to compact the file ────────────────────────────────────────
        log("Running VACUUM …")
        conn.execute("VACUUM")

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
