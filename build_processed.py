"""
build_processed.py
------------------
Builds clean, cross-source, analysis-ready datasets from raw collector output.

Run after each collection cycle:
    python build_processed.py

Output files  (data/processed/)
--------------------------------
match_crossref.csv   Links all three source match IDs via (date, home_team, away_team).
                     This is the keystone — every other processed file joins through it.

shots.csv            Understat shot-level data.
                     Coords normalized to [0,100]. Canonical team names.
                     Crossref IDs (espn_game_id, whoscored_game_id) attached.

events.csv           WhoScored full event stream.
                     Canonical team names. Crossref IDs attached.
                     Suitable for heatmaps, pressure maps, pass networks.

match_summary.csv    One row per match. Understat xG + forecasts + ESPN team stats.
                     Central table for match-level analysis and predictive modelling.

lineups.csv          ESPN lineup data. Canonical team names. Crossref IDs attached.

player_season.csv    Understat player season stats. Canonical team names.

Season config — update these each season (keep in sync with run_collection.py):
"""

import sys
import pandas as pd
from pathlib import Path

from src.sanitize import canonicalize_teams, normalize_coords, normalize_date
from src.utils import ensure_dir

# ---------------------------------------------------------------------------
# Season config — update each season
# ---------------------------------------------------------------------------

US_SEASON   = 2025               # Understat: int start year
ESPN_SEASON = "2025"             # ESPN: string
WS_SEASON   = "2025"             # WhoScored: string
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
# Load
# ---------------------------------------------------------------------------

def load_raw() -> dict[str, pd.DataFrame]:
    """Load all raw data files. Warns and skips any that are missing."""
    sources = {
        "us_matches":    RAW_US   / f"{US_LEAGUE}_{US_SEASON}_matches.csv",
        "us_shots":      RAW_US   / f"{US_LEAGUE}_{US_SEASON}_shots.csv",
        "us_players":    RAW_US   / f"{US_LEAGUE}_{US_SEASON}_players.csv",
        "espn_schedule": RAW_ESPN / f"{ESPN_LEAGUE}_{ESPN_SEASON}_team_schedule.csv",
        "espn_stats":    RAW_ESPN / f"{ESPN_LEAGUE}_{ESPN_SEASON}_match_stats.csv",
        "espn_lineups":  RAW_ESPN / f"{ESPN_LEAGUE}_{ESPN_SEASON}_team_lineups.csv",
        "ws_events":     RAW_WS   / f"{WS_LEAGUE}_{WS_SEASON}_events.csv",
        "ws_schedule":   RAW_WS   / f"{WS_LEAGUE}_{WS_SEASON}_schedule.csv",
    }
    data = {}
    for name, path in sources.items():
        if not path.exists():
            print(f"  ⚠ Missing: {path} — {name} will be skipped")
            data[name] = pd.DataFrame()
        else:
            data[name] = pd.read_csv(path, low_memory=False)
            print(f"  {name:<18} {len(data[name]):>6} rows   ({path.name})")
    return data


# ---------------------------------------------------------------------------
# 1. Match crossref
# ---------------------------------------------------------------------------

def build_match_crossref(data: dict) -> pd.DataFrame:
    """
    Build a table that links Understat, ESPN, and WhoScored match IDs.

    Join key: (match_date as date, canonical home_team, canonical away_team).
    Understat is the spine (380 rows for a full PL season).
    ESPN and WhoScored IDs are left-joined in.
    """
    # --- Understat (spine) ---
    us = data["us_matches"].copy()
    if us.empty:
        print("  ⚠ Understat matches missing — cannot build crossref")
        return pd.DataFrame()

    canonicalize_teams(us, "home_team", "away_team")
    us["match_date"] = normalize_date(us["date"]).dt.date
    us_key = us[[
        "match_date", "home_team", "away_team",
        "match_id", "home_goals", "away_goals",
        "home_xg", "away_xg", "forecast_win", "forecast_draw", "forecast_loss",
    ]].rename(columns={"match_id": "understat_match_id"})

    # --- ESPN ---
    espn_key = pd.DataFrame()
    if not data["espn_schedule"].empty:
        espn = data["espn_schedule"].copy()
        canonicalize_teams(espn, "home_team", "away_team")
        espn["match_date"] = normalize_date(espn["date"]).dt.date
        espn_key = espn[["match_date", "home_team", "away_team", "game_id", "game"]].rename(
            columns={"game_id": "espn_game_id", "game": "espn_game_str"}
        )

    # --- WhoScored ---
    ws_key = pd.DataFrame()
    if not data["ws_schedule"].empty:
        ws = data["ws_schedule"].copy()
        canonicalize_teams(ws, "home_team", "away_team")
        ws["match_date"] = normalize_date(ws["date"]).dt.date
        ws_key = ws[["match_date", "home_team", "away_team", "game_id"]].rename(
            columns={"game_id": "whoscored_game_id"}
        )

    # --- Join ---
    crossref = us_key.copy()
    if not espn_key.empty:
        crossref = crossref.merge(espn_key, on=["match_date", "home_team", "away_team"], how="left")
    else:
        crossref["espn_game_id"] = pd.NA
        crossref["espn_game_str"] = pd.NA

    if not ws_key.empty:
        crossref = crossref.merge(ws_key, on=["match_date", "home_team", "away_team"], how="left")
    else:
        crossref["whoscored_game_id"] = pd.NA

    n = len(crossref)
    n_espn = crossref["espn_game_id"].notna().sum()
    n_ws   = crossref["whoscored_game_id"].notna().sum()
    print(f"  Matches:             {n}")
    print(f"  Linked to ESPN:      {n_espn}/{n} ({100*n_espn//n}%)")
    print(f"  Linked to WhoScored: {n_ws}/{n} ({100*n_ws//n}%)")

    today = pd.Timestamp.now().date()
    unmatched = crossref[crossref["espn_game_id"].isna() | crossref["whoscored_game_id"].isna()].copy()
    if not unmatched.empty:
        future   = unmatched[pd.to_datetime(unmatched["match_date"]).dt.date > today]
        past_gap = unmatched[pd.to_datetime(unmatched["match_date"]).dt.date <= today]
        if not future.empty:
            print(f"  Unmatched (future, expected): {len(future)}")
        if not past_gap.empty:
            print(f"  ⚠ Unmatched (past — may need re-collection): {len(past_gap)}")
            for _, r in past_gap.iterrows():
                missing = []
                if pd.isna(r.get("espn_game_id")):      missing.append("ESPN")
                if pd.isna(r.get("whoscored_game_id")): missing.append("WhoScored")
                print(f"    {r['match_date']}  {r['home_team']} v {r['away_team']}  [{', '.join(missing)}]")

    return crossref.sort_values(["match_date", "home_team"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Shots
# ---------------------------------------------------------------------------

def build_shots(data: dict, crossref: pd.DataFrame) -> pd.DataFrame:
    """
    Understat shot-level data with:
      - Coords normalized from [0,1] → [0,100]
      - Canonical team name derived from home/away side
      - Crossref IDs (espn_game_id, whoscored_game_id) attached
    """
    shots = data["us_shots"].copy()
    if shots.empty:
        return shots

    shots = normalize_coords(shots, source="understat")

    # Attach crossref info
    xref_cols = ["understat_match_id", "home_team", "away_team",
                 "match_date", "espn_game_id", "whoscored_game_id"]
    available = [c for c in xref_cols if c in crossref.columns]
    shots = shots.merge(
        crossref[available],
        left_on="match_id",
        right_on="understat_match_id",
        how="left",
    ).drop(columns=["understat_match_id"], errors="ignore")

    # Canonical team for this shot (home or away)
    shots["team"]     = shots.apply(lambda r: r["home_team"] if r["side"] == "home" else r["away_team"], axis=1)
    shots["opponent"] = shots.apply(lambda r: r["away_team"] if r["side"] == "home" else r["home_team"], axis=1)

    col_order = [
        "match_id", "espn_game_id", "whoscored_game_id",
        "match_date", "team", "opponent", "side",
        "player", "player_id", "minute",
        "x", "y", "xg", "result", "shot_type", "situation",
    ]
    return shots[[c for c in col_order if c in shots.columns]]


# ---------------------------------------------------------------------------
# 3. Events
# ---------------------------------------------------------------------------

def build_events(data: dict, crossref: pd.DataFrame) -> pd.DataFrame:
    """
    WhoScored full event stream with canonical team names and crossref IDs.
    The 'qualifiers' column (nested JSON-like strings) is kept raw — parse
    it in your analysis code as needed.
    """
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

    return events


# ---------------------------------------------------------------------------
# 4. Match summary
# ---------------------------------------------------------------------------

def build_match_summary(data: dict, crossref: pd.DataFrame) -> pd.DataFrame:
    """
    One row per match.

    Columns come from three sources:
      - Understat: xG, goals, forecasts  (always present)
      - ESPN:      possession, shots, passes, etc. (per-team stats pivoted wide)
      - Spine:     crossref IDs, date, teams

    ESPN stats join requires the 'game' column, which is only present in
    ESPN match_stats files generated after the collector was updated to save
    with reset_index(). If missing, the summary still contains all Understat
    data — re-run pull_espn_data() to populate the ESPN columns.
    """
    summary = crossref.copy()

    espn_stats = data["espn_stats"].copy()

    if espn_stats.empty or "game" not in espn_stats.columns:
        print("  ⚠ ESPN match stats missing 'game' column.")
        print("    Re-run pull_espn_data() to generate the updated format,")
        print("    then re-run build_processed.py to fill ESPN columns.")
        return summary

    # Drop the embedded roster JSON — it's large and lives in espn_lineups instead
    if "roster" in espn_stats.columns:
        espn_stats = espn_stats.drop(columns=["roster"])

    # Attach espn_game_id via the schedule's game-string → game_id mapping
    espn_sched = data["espn_schedule"]
    if not espn_sched.empty and "game" in espn_sched.columns:
        game_to_id = espn_sched.set_index("game")["game_id"].to_dict()
        espn_stats["espn_game_id"] = espn_stats["game"].map(game_to_id)
    else:
        return summary

    # Pivot to wide: one row per match with home_ and away_ prefixed columns
    home = espn_stats[espn_stats["is_home"] == True].copy()
    away = espn_stats[espn_stats["is_home"] == False].copy()

    stat_cols = [c for c in espn_stats.columns
                 if c not in ("game", "is_home", "venue", "espn_game_id")]

    home_wide = (home[["espn_game_id", "venue"] + stat_cols]
                 .rename(columns={c: f"home_{c}" for c in stat_cols}))
    away_wide = (away[["espn_game_id"] + stat_cols]
                 .rename(columns={c: f"away_{c}" for c in stat_cols}))

    espn_wide = home_wide.merge(away_wide, on="espn_game_id", how="outer")
    summary = summary.merge(espn_wide, on="espn_game_id", how="left")

    espn_matched = summary["venue"].notna().sum()
    print(f"  ESPN stats joined for {espn_matched}/{len(summary)} matches")

    return summary


# ---------------------------------------------------------------------------
# 5. Lineups
# ---------------------------------------------------------------------------

def build_lineups(data: dict, crossref: pd.DataFrame) -> pd.DataFrame:
    """
    ESPN lineup data with canonical team names and all three source match IDs.
    """
    lineups = data["espn_lineups"].copy()
    if lineups.empty:
        return lineups

    if "roster" in lineups.columns:
        lineups = lineups.drop(columns=["roster"])

    canonicalize_teams(lineups, "team")

    # Map ESPN game string → espn_game_id, then join crossref
    espn_sched = data["espn_schedule"]
    if not espn_sched.empty and "game" in espn_sched.columns and "game" in lineups.columns:
        game_to_id = espn_sched.set_index("game")["game_id"].to_dict()
        lineups["espn_game_id"] = lineups["game"].map(game_to_id)

        xref_ids = crossref[["espn_game_id", "understat_match_id",
                              "whoscored_game_id", "match_date"]].copy()
        lineups = lineups.merge(xref_ids, on="espn_game_id", how="left")

    return lineups


# ---------------------------------------------------------------------------
# 6. Player season stats
# ---------------------------------------------------------------------------

def build_player_season(data: dict) -> pd.DataFrame:
    """
    Understat player season stats with canonical team names.

    Understat uses comma-separated team strings for mid-season transfers
    (e.g. "Bournemouth,Manchester City"). The 'team' column is kept as-is
    (after canonicalization) for full history. A 'primary_team' column is
    added with the player's most recent club (last in the list).
    """
    players = data["us_players"].copy()
    if players.empty:
        return players

    canonicalize_teams(players, "team")

    # primary_team = last team in the (potentially comma-separated) string
    players["primary_team"] = players["team"].apply(
        lambda t: str(t).split(",")[-1].strip() if pd.notna(t) else t
    )

    # Reorder: put primary_team next to team
    cols = list(players.columns)
    team_idx = cols.index("team")
    cols.insert(team_idx + 1, cols.pop(cols.index("primary_team")))
    return players[cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dir(str(OUT))

    print("=" * 55)
    print("build_processed.py")
    print(f"Season: Understat={US_SEASON}  ESPN={ESPN_SEASON}  WhoScored={WS_SEASON}")
    print("=" * 55)

    print("\n[1/7] Loading raw data...")
    data = load_raw()

    print("\n[2/7] Building match crossref...")
    crossref = build_match_crossref(data)

    print("\n[3/7] Building shots...")
    shots = build_shots(data, crossref)

    print("\n[4/7] Building events...")
    events = build_events(data, crossref)

    print("\n[5/7] Building match summary...")
    match_summary = build_match_summary(data, crossref)

    print("\n[6/7] Building lineups...")
    lineups = build_lineups(data, crossref)

    print("\n[7/7] Building player season stats...")
    player_season = build_player_season(data)

    print("\nSaving to data/processed/ ...")
    outputs = {
        "match_crossref.csv": crossref,
        "shots.csv":          shots,
        "events.csv":         events,
        "match_summary.csv":  match_summary,
        "lineups.csv":        lineups,
        "player_season.csv":  player_season,
    }
    for filename, df in outputs.items():
        path = OUT / filename
        df.to_csv(path, index=False)
        print(f"  ✓ {filename:<25} {len(df):>7} rows")

    print("\n✓ Done. All files saved to data/processed/")


if __name__ == "__main__":
    main()
