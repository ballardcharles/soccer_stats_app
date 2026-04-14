"""
build_processed.py
------------------
Builds clean, cross-source, analysis-ready datasets from raw collector output.

Automatically detects every season that has an Understat matches file in
data/raw_understat/ and processes them all.  Add a new season simply by
running the collectors for it — no config change needed here.

Run after each collection cycle:
    python build_processed.py

Output files  (data/processed/)
--------------------------------
match_crossref.csv   Links all three source match IDs via (date, home_team, away_team).
shots.csv            Understat shot-level data. Coords in [0,100]. Canonical team names.
events.csv           WhoScored full event stream with crossref IDs.
match_summary.csv    One row per match. Understat xG + forecasts + ESPN team stats.
lineups.csv          ESPN lineup data. Canonical team names. Crossref IDs attached.
player_season.csv    Understat player season stats. Canonical team names.

All output files contain a 'season' column (e.g. '2024/25') so the dashboard
can filter by season without needing separate files per year.
"""

import re
import sys
import pandas as pd
from pathlib import Path

from src.sanitize import canonicalize_teams, normalize_coords, normalize_date
from src.utils import ensure_dir

# ---------------------------------------------------------------------------
# League identifiers (fixed — only the year changes each season)
# ---------------------------------------------------------------------------

US_LEAGUE   = "EPL"
ESPN_LEAGUE = "ENG-Premier League"
WS_LEAGUE   = "ENG-Premier League"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_US   = Path("data/raw_understat")
RAW_ESPN = Path("data/raw_espn")
RAW_WS   = Path("data/raw_whoscored")
OUT      = Path("data/processed")


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------

def season_label(us_year: int) -> str:
    """Convert an Understat year integer to a display label.

    The Understat year represents the END calendar year of the season.
    Examples:
        2025  ->  '2024/25'
        2024  ->  '2023/24'
        2023  ->  '2022/23'
    """
    return f"{us_year - 1}/{str(us_year)[2:]}"


def detect_seasons() -> list[int]:
    """Scan raw_understat/ for all available seasons.

    Looks for files matching 'EPL_{year}_matches.csv' and returns the
    sorted list of year integers found.  This means build_processed.py
    automatically picks up new seasons as soon as the collector has run —
    no manual config update required.
    """
    pattern = re.compile(r"EPL_(\d{4})_matches\.csv")
    seasons = []
    for f in RAW_US.glob("EPL_*_matches.csv"):
        m = pattern.match(f.name)
        if m:
            seasons.append(int(m.group(1)))
    if not seasons:
        print("  No Understat match files found in data/raw_understat/")
        print("  Run `python run_collection.py` first.")
    return sorted(seasons)


# ---------------------------------------------------------------------------
# Load raw data for a single season
# ---------------------------------------------------------------------------

def load_raw(us_season: int) -> dict[str, pd.DataFrame]:
    """Load all raw source files for one season.

    Missing files are returned as empty DataFrames — build functions
    downstream handle absent data gracefully.
    """
    espn_season = str(us_season)
    ws_season   = str(us_season)

    sources = {
        "us_matches":    RAW_US   / f"{US_LEAGUE}_{us_season}_matches.csv",
        "us_shots":      RAW_US   / f"{US_LEAGUE}_{us_season}_shots.csv",
        "us_players":    RAW_US   / f"{US_LEAGUE}_{us_season}_players.csv",
        "espn_schedule": RAW_ESPN / f"{ESPN_LEAGUE}_{espn_season}_team_schedule.csv",
        "espn_stats":    RAW_ESPN / f"{ESPN_LEAGUE}_{espn_season}_match_stats.csv",
        "espn_lineups":  RAW_ESPN / f"{ESPN_LEAGUE}_{espn_season}_team_lineups.csv",
        "ws_events":     RAW_WS   / f"{WS_LEAGUE}_{ws_season}_events.csv",
        "ws_schedule":   RAW_WS   / f"{WS_LEAGUE}_{ws_season}_schedule.csv",
    }
    data = {}
    for name, path in sources.items():
        if not path.exists():
            data[name] = pd.DataFrame()
        else:
            data[name] = pd.read_csv(path, low_memory=False)
    return data


# ---------------------------------------------------------------------------
# 1. Match crossref
# ---------------------------------------------------------------------------

def build_match_crossref(data: dict, us_season: int) -> pd.DataFrame:
    """
    Build a table linking Understat, ESPN, and WhoScored match IDs.

    Join key: (match_date as date, canonical home_team, canonical away_team).
    Understat is the spine. ESPN and WhoScored IDs are left-joined in.
    A 'season' column is added so rows from all seasons can be combined.
    """
    us = data["us_matches"].copy()
    if us.empty:
        return pd.DataFrame()

    canonicalize_teams(us, "home_team", "away_team")
    us["match_date"] = normalize_date(us["date"]).dt.date
    us_key = us[[
        "match_date", "home_team", "away_team",
        "match_id", "home_goals", "away_goals",
        "home_xg", "away_xg", "forecast_win", "forecast_draw", "forecast_loss",
    ]].rename(columns={"match_id": "understat_match_id"})

    # ESPN
    espn_key = pd.DataFrame()
    if not data["espn_schedule"].empty:
        espn = data["espn_schedule"].copy()
        canonicalize_teams(espn, "home_team", "away_team")
        espn["match_date"] = normalize_date(espn["date"]).dt.date
        espn_key = espn[["match_date", "home_team", "away_team", "game_id", "game"]].rename(
            columns={"game_id": "espn_game_id", "game": "espn_game_str"}
        )

    # WhoScored
    ws_key = pd.DataFrame()
    if not data["ws_schedule"].empty:
        ws = data["ws_schedule"].copy()
        canonicalize_teams(ws, "home_team", "away_team")
        ws["match_date"] = normalize_date(ws["date"]).dt.date
        ws_key = ws[["match_date", "home_team", "away_team", "game_id"]].rename(
            columns={"game_id": "whoscored_game_id"}
        )

    # Join
    crossref = us_key.copy()
    if not espn_key.empty:
        crossref = crossref.merge(espn_key, on=["match_date", "home_team", "away_team"], how="left")
    else:
        crossref["espn_game_id"]  = pd.NA
        crossref["espn_game_str"] = pd.NA

    if not ws_key.empty:
        crossref = crossref.merge(ws_key, on=["match_date", "home_team", "away_team"], how="left")
    else:
        crossref["whoscored_game_id"] = pd.NA

    crossref["season"] = season_label(us_season)

    n      = len(crossref)
    n_espn = crossref["espn_game_id"].notna().sum()
    n_ws   = crossref["whoscored_game_id"].notna().sum()
    pct_e  = (100 * n_espn // n) if n else 0
    pct_w  = (100 * n_ws   // n) if n else 0
    print(f"    {n} matches  |  ESPN {n_espn}/{n} ({pct_e}%)  |  WhoScored {n_ws}/{n} ({pct_w}%)")

    today    = pd.Timestamp.now().date()
    past_gap = crossref[
        (crossref["espn_game_id"].isna() | crossref["whoscored_game_id"].isna()) &
        (pd.to_datetime(crossref["match_date"]).dt.date <= today)
    ]
    if not past_gap.empty:
        print(f"    ⚠  {len(past_gap)} past match(es) missing source links")

    return crossref.sort_values(["match_date", "home_team"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Shots
# ---------------------------------------------------------------------------

def build_shots(data: dict, crossref: pd.DataFrame, us_season: int) -> pd.DataFrame:
    """Understat shot-level data with normalised coords and crossref IDs."""
    shots = data["us_shots"].copy()
    if shots.empty:
        return shots

    shots = normalize_coords(shots, source="understat")

    xref_cols = ["understat_match_id", "home_team", "away_team",
                 "match_date", "espn_game_id", "whoscored_game_id"]
    available = [c for c in xref_cols if c in crossref.columns]
    shots = shots.merge(
        crossref[available],
        left_on="match_id",
        right_on="understat_match_id",
        how="left",
    ).drop(columns=["understat_match_id"], errors="ignore")

    shots["team"]     = shots.apply(lambda r: r["home_team"] if r["side"] == "home" else r["away_team"], axis=1)
    shots["opponent"] = shots.apply(lambda r: r["away_team"] if r["side"] == "home" else r["home_team"], axis=1)
    shots["season"]   = season_label(us_season)

    col_order = [
        "season", "match_id", "espn_game_id", "whoscored_game_id",
        "match_date", "team", "opponent", "side",
        "player", "player_id", "minute",
        "x", "y", "xg", "result", "shot_type", "situation",
    ]
    return shots[[c for c in col_order if c in shots.columns]]


# ---------------------------------------------------------------------------
# 3. Events
# ---------------------------------------------------------------------------

def build_events(data: dict, crossref: pd.DataFrame, us_season: int) -> pd.DataFrame:
    """WhoScored full event stream with canonical team names and crossref IDs."""
    events = data["ws_events"].copy()
    if events.empty:
        return events

    canonicalize_teams(events, "team")

    if "whoscored_game_id" in crossref.columns:
        xref = (
            crossref[["whoscored_game_id", "understat_match_id", "espn_game_id",
                       "home_team", "away_team", "match_date"]]
            .dropna(subset=["whoscored_game_id"])
            .copy()
        )
        xref["whoscored_game_id"] = xref["whoscored_game_id"].astype("Int64")
        events["game_id"] = events["game_id"].astype("Int64")
        events = events.merge(xref, left_on="game_id", right_on="whoscored_game_id", how="left")

    events["season"] = season_label(us_season)
    return events


# ---------------------------------------------------------------------------
# 4. Match summary
# ---------------------------------------------------------------------------

def build_match_summary(data: dict, crossref: pd.DataFrame, us_season: int) -> pd.DataFrame:
    """One row per match: Understat xG + forecasts + ESPN team stats (if available)."""
    summary = crossref.copy()

    espn_stats = data["espn_stats"].copy()

    if espn_stats.empty or "game" not in espn_stats.columns:
        return summary

    if "roster" in espn_stats.columns:
        espn_stats = espn_stats.drop(columns=["roster"])

    espn_sched = data["espn_schedule"]
    if not espn_sched.empty and "game" in espn_sched.columns:
        game_to_id = espn_sched.set_index("game")["game_id"].to_dict()
        espn_stats["espn_game_id"] = espn_stats["game"].map(game_to_id)
    else:
        return summary

    home = espn_stats[espn_stats["is_home"] == True].copy()
    away = espn_stats[espn_stats["is_home"] == False].copy()

    stat_cols = [c for c in espn_stats.columns
                 if c not in ("game", "is_home", "venue", "espn_game_id")]

    home_wide = (home[["espn_game_id", "venue"] + stat_cols]
                 .rename(columns={c: f"home_{c}" for c in stat_cols}))
    away_wide = (away[["espn_game_id"] + stat_cols]
                 .rename(columns={c: f"away_{c}" for c in stat_cols}))

    espn_wide = home_wide.merge(away_wide, on="espn_game_id", how="outer")
    summary   = summary.merge(espn_wide, on="espn_game_id", how="left")

    return summary


# ---------------------------------------------------------------------------
# 5. Lineups
# ---------------------------------------------------------------------------

def build_lineups(data: dict, crossref: pd.DataFrame, us_season: int) -> pd.DataFrame:
    """ESPN lineup data with canonical team names and all three source match IDs."""
    lineups = data["espn_lineups"].copy()
    if lineups.empty:
        return lineups

    if "roster" in lineups.columns:
        lineups = lineups.drop(columns=["roster"])

    canonicalize_teams(lineups, "team")

    espn_sched = data["espn_schedule"]
    if not espn_sched.empty and "game" in espn_sched.columns and "game" in lineups.columns:
        game_to_id = espn_sched.set_index("game")["game_id"].to_dict()
        lineups["espn_game_id"] = lineups["game"].map(game_to_id)

        xref_ids = crossref[["espn_game_id", "understat_match_id",
                              "whoscored_game_id", "match_date"]].copy()
        lineups = lineups.merge(xref_ids, on="espn_game_id", how="left")

    lineups["season"] = season_label(us_season)
    return lineups


# ---------------------------------------------------------------------------
# 6. Player season stats
# ---------------------------------------------------------------------------

def build_player_season(data: dict, us_season: int) -> pd.DataFrame:
    """Understat player season stats with canonical team names and season label."""
    players = data["us_players"].copy()
    if players.empty:
        return players

    canonicalize_teams(players, "team")

    players["primary_team"] = players["team"].apply(
        lambda t: str(t).split(",")[-1].strip() if pd.notna(t) else t
    )

    cols = list(players.columns)
    team_idx = cols.index("team")
    cols.insert(team_idx + 1, cols.pop(cols.index("primary_team")))
    players = players[cols]

    players["season"] = season_label(us_season)
    return players


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dir(str(OUT))

    seasons = detect_seasons()
    if not seasons:
        sys.exit(1)

    print("=" * 60)
    print("build_processed.py  —  multi-season build")
    print(f"Seasons found: {[season_label(s) for s in seasons]}")
    print("=" * 60)

    all_crossref = []
    all_shots    = []
    all_events   = []
    all_summary  = []
    all_lineups  = []
    all_players  = []

    for us_season in seasons:
        lbl = season_label(us_season)
        print(f"\n── {lbl} (year={us_season}) ──")

        data = load_raw(us_season)

        missing = [k for k, v in data.items() if v.empty]
        if missing:
            print(f"  Missing source files: {', '.join(missing)}")

        if data["us_matches"].empty:
            print(f"  Skipping {lbl} — no Understat matches file")
            continue

        crossref = build_match_crossref(data, us_season)
        all_crossref.append(crossref)

        shots = build_shots(data, crossref, us_season)
        if not shots.empty: all_shots.append(shots)

        events = build_events(data, crossref, us_season)
        if not events.empty: all_events.append(events)

        summary = build_match_summary(data, crossref, us_season)
        all_summary.append(summary)

        lineups = build_lineups(data, crossref, us_season)
        if not lineups.empty: all_lineups.append(lineups)

        players = build_player_season(data, us_season)
        if not players.empty: all_players.append(players)

    # Concatenate all seasons and write to disk
    print(f"\n{'='*60}")
    print("Saving combined output to data/processed/ ...")

    def safe_concat(frames: list) -> pd.DataFrame:
        frames = [f for f in frames if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    outputs = {
        "match_crossref.csv": safe_concat(all_crossref),
        "shots.csv":          safe_concat(all_shots),
        "events.csv":         safe_concat(all_events),
        "match_summary.csv":  safe_concat(all_summary),
        "lineups.csv":        safe_concat(all_lineups),
        "player_season.csv":  safe_concat(all_players),
    }

    for filename, df in outputs.items():
        path = OUT / filename
        df.to_csv(path, index=False)
        season_list = sorted(df["season"].unique()) if ("season" in df.columns and not df.empty) else []
        print(f"  ✓ {filename:<25} {len(df):>8} rows   {season_list}")

    print("\n✓ Done. All files saved to data/processed/")


if __name__ == "__main__":
    main()
