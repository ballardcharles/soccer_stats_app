"""
scoring.py
==========
Computes 1-10 grades for teams (per season and rolling per-match form) and
players (per season) from processed soccer match data.

Grade philosophy
----------------
All grades use min-max scaling within a season so that a 10 means "best in
that season's competition" and a 1 means "worst".  Missing ESPN stat columns
(e.g. tackles, possession) are median-imputed before scaling so one absent
match does not poison an entire team's grade.

Entry points
------------
flatten_match_summary(ms)        wide match_summary  ->  long (2 rows/match)
compute_season_grades(flat)      long format         ->  1 row per team×season
compute_rolling_grades(flat, n)  long format         ->  1 row per team×match
compute_player_grades(ps)        player_season CSV   ->  1 row per player×season
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _minmax_scale(series: pd.Series, invert: bool = False) -> pd.Series:
    """Scale a numeric Series to the range [1.0, 10.0].

    Parameters
    ----------
    series : pd.Series
        Raw numeric values (NaNs allowed; they map to NaN in output).
    invert : bool
        If True, a *lower* raw value produces a *higher* grade (e.g. goals
        conceded: fewer is better).

    Returns
    -------
    pd.Series
        Grades in [1.0, 10.0].  If all values are identical (min == max) every
        grade is set to the midpoint 5.5 to avoid division-by-zero.
    """
    lo = series.min()
    hi = series.max()

    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        # No meaningful spread — assign neutral midpoint
        return pd.Series(5.5, index=series.index, dtype=float)

    if invert:
        scaled = (hi - series) / (hi - lo)
    else:
        scaled = (series - lo) / (hi - lo)

    # Stretch [0, 1] → [1, 10]
    return scaled * 9.0 + 1.0


# ---------------------------------------------------------------------------
# 1.  Flatten wide match_summary → long format
# ---------------------------------------------------------------------------

def flatten_match_summary(ms: pd.DataFrame) -> pd.DataFrame:
    """Convert wide match_summary (one row per match) to long format.

    Each match produces two rows — one for the home team and one for the away
    team.  Derived columns (shots_on_target_pct, xg_per_shot, result, points)
    are computed here so every downstream function can rely on them.

    Parameters
    ----------
    ms : pd.DataFrame
        Raw match_summary.csv loaded as a DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns:
          season, match_date, team, opponent, is_home,
          goals, goals_conceded, xg, xga,
          shots, shots_on_target, shots_on_target_pct, xg_per_shot,
          possession_pct, pass_pct, tackle_pct, interceptions,
          saves, fouls, yellow_cards, red_cards,
          result (W/D/L), points (3/1/0)
    """
    # Parse match_date to timezone-aware datetime; bad values become NaT
    ms = ms.copy()
    ms["match_date"] = pd.to_datetime(ms["match_date"], utc=True, errors="coerce")

    def _col(src: pd.DataFrame, name: str) -> pd.Series:
        """Return column or an all-NaN Series if the column is absent."""
        return src[name] if name in src.columns else pd.Series(np.nan, index=src.index)

    def _build_side(src: pd.DataFrame, mine: str, theirs: str) -> pd.DataFrame:
        """Vectorised construction of one long-format row per match for one side."""
        g_me  = pd.to_numeric(src[f"{mine}_goals"],   errors="coerce")
        g_opp = pd.to_numeric(src[f"{theirs}_goals"],  errors="coerce")
        xg_me = pd.to_numeric(src[f"{mine}_xg"],      errors="coerce")
        shots  = pd.to_numeric(_col(src, f"{mine}_total_shots"),       errors="coerce")
        sot    = pd.to_numeric(_col(src, f"{mine}_shots_on_target"),   errors="coerce")

        both_known = g_me.notna() & g_opp.notna()
        result = np.where(~both_known, None,
                 np.where(g_me > g_opp, "W",
                 np.where(g_me == g_opp, "D", "L")))
        points = np.where(result == "W", 3.0,
                 np.where(result == "D", 1.0,
                 np.where(result == "L", 0.0, np.nan)))

        shots_pos = shots.fillna(0) > 0
        sot_pct     = np.where(shots_pos, sot / shots * 100, np.nan)
        xg_per_shot = np.where(shots_pos, xg_me / shots,     np.nan)

        return pd.DataFrame({
            "season":              src["season"],
            "match_date":          src["match_date"],
            "team":                src[f"{mine}_team"],
            "opponent":            src[f"{theirs}_team"],
            "is_home":             (mine == "home"),
            "goals":               g_me,
            "goals_conceded":      g_opp,
            "xg":                  xg_me,
            "xga":                 pd.to_numeric(src[f"{theirs}_xg"], errors="coerce"),
            "shots":               shots,
            "shots_on_target":     sot,
            "shots_on_target_pct": sot_pct,
            "xg_per_shot":         xg_per_shot,
            "possession_pct":      pd.to_numeric(_col(src, f"{mine}_possession_pct"),    errors="coerce"),
            "pass_pct":            pd.to_numeric(_col(src, f"{mine}_pass_pct"),          errors="coerce"),
            "tackle_pct":          pd.to_numeric(_col(src, f"{mine}_tackle_pct"),        errors="coerce"),
            "interceptions":       pd.to_numeric(_col(src, f"{mine}_interceptions"),     errors="coerce"),
            "saves":               pd.to_numeric(_col(src, f"{mine}_saves"),             errors="coerce"),
            "fouls":               pd.to_numeric(_col(src, f"{mine}_fouls_committed"),   errors="coerce"),
            "yellow_cards":        pd.to_numeric(_col(src, f"{mine}_yellow_cards"),      errors="coerce"),
            "red_cards":           pd.to_numeric(_col(src, f"{mine}_red_cards"),         errors="coerce"),
            "result":              pd.array(result, dtype=object),
            "points":              pd.to_numeric(pd.array(points, dtype=object), errors="coerce"),
        })

    flat = pd.concat(
        [_build_side(ms, "home", "away"), _build_side(ms, "away", "home")],
        ignore_index=True,
    )
    flat = flat.sort_values(["season", "match_date", "team"]).reset_index(drop=True)
    return flat


# ---------------------------------------------------------------------------
# 2.  Season-level team grades
# ---------------------------------------------------------------------------

# Metric columns used in grading (subject to median imputation)
_GRADE_METRICS = [
    "xg", "goals", "shots_on_target_pct", "xg_per_shot",   # attack
    "xga", "goals_conceded", "saves", "interceptions",      # defense
    "possession_pct", "pass_pct", "tackle_pct",             # style
]


def _impute_medians(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Fill NaN values in *cols* with the column-wide median.

    This prevents missing ESPN stat columns from propagating NaN grades while
    keeping the imputation conservative (median, not mean).
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def compute_season_grades(flat: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate 1-10 grades for every team × season.

    Steps
    -----
    1. Aggregate per-match metrics to per-team averages within each season.
    2. Impute missing averages with the season-level column median.
    3. Scale each averaged metric to [1, 10] using min-max within the season
       (so grades reflect standing relative to same-season competition).
    4. Combine sub-grades using the weighted formulas below.

    Grade formulas
    --------------
    attack_grade  = 0.40×xg + 0.20×goals + 0.20×shots_on_target_pct + 0.20×xg_per_shot
    defense_grade = 0.40×(xga inv) + 0.20×(goals_conceded inv) + 0.20×saves + 0.20×interceptions
    style_grade   = 0.40×possession_pct + 0.40×pass_pct + 0.20×tackle_pct
    overall_grade = 0.40×attack + 0.40×defense + 0.20×style

    Parameters
    ----------
    flat : pd.DataFrame
        Output of flatten_match_summary().

    Returns
    -------
    pd.DataFrame
        One row per team × season.
    """
    # --- Aggregate raw metrics to team×season means -----------------------
    agg_cols = _GRADE_METRICS + ["points", "result"]

    agg = (
        flat.groupby(["season", "team"])
        .agg(
            matches_played=("goals", "count"),
            wins=("result", lambda x: (x == "W").sum()),
            draws=("result", lambda x: (x == "D").sum()),
            losses=("result", lambda x: (x == "L").sum()),
            avg_xg=("xg", "mean"),
            avg_xga=("xga", "mean"),
            avg_goals=("goals", "mean"),
            avg_goals_conceded=("goals_conceded", "mean"),
            avg_points=("points", "mean"),
            avg_shots_on_target_pct=("shots_on_target_pct", "mean"),
            avg_xg_per_shot=("xg_per_shot", "mean"),
            avg_saves=("saves", "mean"),
            avg_interceptions=("interceptions", "mean"),
            avg_possession_pct=("possession_pct", "mean"),
            avg_pass_pct=("pass_pct", "mean"),
            avg_tackle_pct=("tackle_pct", "mean"),
        )
        .reset_index()
    )

    # --- Scale metrics to [1,10] within each season -----------------------
    grade_rows = []

    for season, group in agg.groupby("season"):
        g = group.copy()

        # Median-impute averaged metrics before scaling
        avg_metric_cols = [
            "avg_xg", "avg_xga", "avg_goals", "avg_goals_conceded",
            "avg_shots_on_target_pct", "avg_xg_per_shot",
            "avg_saves", "avg_interceptions",
            "avg_possession_pct", "avg_pass_pct", "avg_tackle_pct",
        ]
        g = _impute_medians(g, avg_metric_cols)

        # Scale each metric dimension (invert=True → fewer is better)
        sc_xg       = _minmax_scale(g["avg_xg"])
        sc_goals    = _minmax_scale(g["avg_goals"])
        sc_sot_pct  = _minmax_scale(g["avg_shots_on_target_pct"])
        sc_xgps     = _minmax_scale(g["avg_xg_per_shot"])

        sc_xga_inv  = _minmax_scale(g["avg_xga"], invert=True)
        sc_gc_inv   = _minmax_scale(g["avg_goals_conceded"], invert=True)
        sc_saves    = _minmax_scale(g["avg_saves"])
        sc_inter    = _minmax_scale(g["avg_interceptions"])

        sc_poss     = _minmax_scale(g["avg_possession_pct"])
        sc_pass     = _minmax_scale(g["avg_pass_pct"])
        sc_tackle   = _minmax_scale(g["avg_tackle_pct"])

        # Weighted sub-grade composites (weights already sum to 1 for each)
        attack_raw  = 0.40 * sc_xg + 0.20 * sc_goals + 0.20 * sc_sot_pct + 0.20 * sc_xgps
        defense_raw = 0.40 * sc_xga_inv + 0.20 * sc_gc_inv + 0.20 * sc_saves + 0.20 * sc_inter
        style_raw   = 0.40 * sc_poss + 0.40 * sc_pass + 0.20 * sc_tackle

        # Weighted composites of [1,10] inputs (weights sum to 1) are already
        # in [1,10] — re-scaling would distort the documented weights.
        g["attack_grade"]  = attack_raw
        g["defense_grade"] = defense_raw
        g["style_grade"]   = style_raw

        g["overall_grade"] = (
            0.40 * g["attack_grade"]
            + 0.40 * g["defense_grade"]
            + 0.20 * g["style_grade"]
        )

        grade_rows.append(g)

    result = pd.concat(grade_rows, ignore_index=True)

    # Tidy column ordering
    col_order = [
        "season", "team",
        "matches_played", "wins", "draws", "losses", "avg_points",
        "avg_xg", "avg_xga", "avg_goals", "avg_goals_conceded",
        "attack_grade", "defense_grade", "style_grade", "overall_grade",
    ]
    extra = [c for c in result.columns if c not in col_order]
    return result[col_order + extra]


# ---------------------------------------------------------------------------
# 3.  Rolling (per-match form) grades
# ---------------------------------------------------------------------------

def compute_rolling_grades(flat: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Compute rolling n-match form grades for every team × match.

    For each match the rolling average covers the *n* matches immediately
    before it (shift(1) ensures the current match is excluded).  This
    represents the form a team carries *into* each fixture.

    Scaling uses the season-wide distribution of rolling values (not just
    those before the match) so that early-season entries with few prior
    matches are scaled on the same axis as mid/late season.

    Parameters
    ----------
    flat : pd.DataFrame
        Output of flatten_match_summary().
    n : int
        Window size for rolling average (default 5).

    Returns
    -------
    pd.DataFrame
        All columns from *flat* plus:
          roll_xg, roll_xga, roll_points,
          roll_attack_grade, roll_defense_grade,
          roll_style_grade, roll_overall_grade
    """
    roll_metric_cols = [
        "xg", "xga", "goals", "goals_conceded",
        "shots_on_target_pct", "xg_per_shot",
        "saves", "interceptions",
        "possession_pct", "pass_pct", "tackle_pct",
        "points",
    ]

    flat = flat.copy()

    # Median-impute raw metrics before rolling so NaN gaps don't widen windows
    flat = _impute_medians(flat, roll_metric_cols)

    # Sort chronologically within each team
    flat = flat.sort_values(["team", "match_date"]).reset_index(drop=True)

    # Compute rolling means (shift by 1 so current match is not included)
    roll_parts = []
    for team, grp in flat.groupby("team", sort=False):
        grp = grp.copy()
        for col in roll_metric_cols:
            grp[f"roll_{col}"] = (
                grp[col]
                .shift(1)                         # exclude current match
                .rolling(window=n, min_periods=1)  # require at least 1 prior match
                .mean()
            )
        roll_parts.append(grp)

    result = pd.concat(roll_parts).sort_values(["season", "match_date", "team"]).reset_index(drop=True)

    # --- Scale rolling metrics to [1,10] within each season ---------------
    grade_parts = []

    for season, grp in result.groupby("season"):
        g = grp.copy()

        # Median-impute rolling columns (NaN for a team's very first match)
        roll_cols = [f"roll_{c}" for c in roll_metric_cols]
        g = _impute_medians(g, roll_cols)

        sc_xg      = _minmax_scale(g["roll_xg"])
        sc_goals   = _minmax_scale(g["roll_goals"])
        sc_sot_pct = _minmax_scale(g["roll_shots_on_target_pct"])
        sc_xgps    = _minmax_scale(g["roll_xg_per_shot"])

        sc_xga_inv = _minmax_scale(g["roll_xga"], invert=True)
        sc_gc_inv  = _minmax_scale(g["roll_goals_conceded"], invert=True)
        sc_saves   = _minmax_scale(g["roll_saves"])
        sc_inter   = _minmax_scale(g["roll_interceptions"])

        sc_poss    = _minmax_scale(g["roll_possession_pct"])
        sc_pass    = _minmax_scale(g["roll_pass_pct"])
        sc_tackle  = _minmax_scale(g["roll_tackle_pct"])

        attack_raw  = 0.40 * sc_xg + 0.20 * sc_goals + 0.20 * sc_sot_pct + 0.20 * sc_xgps
        defense_raw = 0.40 * sc_xga_inv + 0.20 * sc_gc_inv + 0.20 * sc_saves + 0.20 * sc_inter
        style_raw   = 0.40 * sc_poss + 0.40 * sc_pass + 0.20 * sc_tackle

        g["roll_attack_grade"]  = attack_raw
        g["roll_defense_grade"] = defense_raw
        g["roll_style_grade"]   = style_raw

        g["roll_overall_grade"] = (
            0.40 * g["roll_attack_grade"]
            + 0.40 * g["roll_defense_grade"]
            + 0.20 * g["roll_style_grade"]
        )

        grade_parts.append(g)

    return pd.concat(grade_parts).sort_values(["season", "match_date", "team"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4.  Player season grades
# ---------------------------------------------------------------------------

# Position group mapping: extract the primary position code from Understat's
# multi-character position strings (e.g. "D S" → "D", "GK S" → "GK").
_POS_GROUP = {
    "GK":  "GK",
    "D":   "DEF",
    "D M": "MID",
    "D F": "FWD",
    "M":   "MID",
    "F M": "MID",
    "F":   "FWD",
}

def _pos_group(pos_str: str) -> str:
    """Map an Understat position string to one of: GK / DEF / MID / FWD.

    Understat encodes position as space-separated codes plus an optional
    trailing 'S' for substitute (e.g. "D S", "F M S", "GK S").
    We strip trailing 'S' and match against _POS_GROUP; default is 'MID'.
    """
    if pd.isna(pos_str):
        return "MID"
    # Remove trailing substitute flag and extra whitespace
    clean = str(pos_str).replace(" S", "").strip()
    return _POS_GROUP.get(clean, "MID")


def _aggregate_defensive_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate WhoScored defensive actions per player × team × season.

    Actions counted:
      tackles          — total Tackle events
      tackles_won      — Tackle events with outcome 'Successful'
      interceptions    — Interception events
      clearances       — Clearance events
      ball_recoveries  — BallRecovery events
      aerials          — Aerial events
      aerials_won      — Aerial events with outcome 'Successful'

    Returns
    -------
    pd.DataFrame  — one row per player × team × season.
    """
    # Only rows that are defensive action types
    def_types = {"Tackle", "Interception", "Clearance", "BallRecovery", "Aerial"}
    ev = events_df[events_df["type"].isin(def_types)].copy()

    if ev.empty:
        return pd.DataFrame(columns=[
            "player", "team", "season",
            "tackles", "tackles_won", "interceptions",
            "clearances", "ball_recoveries", "aerials", "aerials_won",
        ])

    # Build boolean helper columns so groupby lambdas stay readable
    ev["_tackle"]      = ev["type"] == "Tackle"
    ev["_tackle_won"]  = (ev["type"] == "Tackle")  & (ev["outcome_type"] == "Successful")
    ev["_intercept"]   = ev["type"] == "Interception"
    ev["_clearance"]   = ev["type"] == "Clearance"
    ev["_recovery"]    = ev["type"] == "BallRecovery"
    ev["_aerial"]      = ev["type"] == "Aerial"
    ev["_aerial_won"]  = (ev["type"] == "Aerial")  & (ev["outcome_type"] == "Successful")

    agg = (
        ev.groupby(["player", "team", "season"])
        .agg(
            tackles=("_tackle",     "sum"),
            tackles_won=("_tackle_won",  "sum"),
            interceptions=("_intercept",  "sum"),
            clearances=("_clearance", "sum"),
            ball_recoveries=("_recovery",  "sum"),
            aerials=("_aerial",    "sum"),
            aerials_won=("_aerial_won",  "sum"),
        )
        .reset_index()
    )
    return agg


def _aggregate_gk_stats(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate goalkeeper saves / shots faced from lineups per player × season.

    Returns
    -------
    pd.DataFrame — one row per GK player × team × season with columns:
        player, team, season, total_saves, total_shots_faced, total_goals_conceded
    """
    gk = lineups_df[
        lineups_df["position"].str.contains("Goalkeeper", na=False)
    ].copy()

    if gk.empty:
        return pd.DataFrame(columns=[
            "player", "team", "season",
            "total_saves", "total_shots_faced", "total_goals_conceded",
        ])

    agg = (
        gk.groupby(["player", "team", "season"])
        .agg(
            total_saves=("saves", "sum"),
            total_shots_faced=("shots_faced", "sum"),
            total_goals_conceded=("goals_conceded", "sum"),
        )
        .reset_index()
    )
    return agg


def compute_player_grades(
    player_season: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    lineups_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute 1-10 grades for players, one row per player × season.

    Only players with at least 90 minutes played are included (sufficient
    sample to compute meaningful per-90 metrics).

    Offensive metrics (from player_season / Understat)
    ---------------------------------------------------
    npxg_p90     = npxg / (time / 90)   — non-penalty xG per 90 min
    xa_p90       = xa   / (time / 90)   — expected assists per 90 min
    kp_p90       = key_passes / (time / 90)
    xg_chain_p90 = xg_chain  / (time / 90)

    Defensive metrics (from events_df / WhoScored)
    -----------------------------------------------
    tackles_won_p90    — successful tackles per 90
    interceptions_p90  — interceptions per 90
    clearances_p90     — clearances per 90
    recoveries_p90     — ball recoveries per 90

    Goalkeeper metrics (from lineups_df / ESPN)
    -------------------------------------------
    saves_p90  — saves per 90 minutes
    Note: ESPN's shots_faced includes all attempts (on and off target) so
    save_pct (saves/shots_faced) is NOT used — it produces unreliable values.
    GK grade is based solely on saves_p90.

    Grade formulas (all scaled within season × position group)
    ----------------------------------------------------------
    attack_grade     = minmax(npxg_p90)
    creativity_grade = minmax(0.6×xa_p90 + 0.4×kp_p90)
    defensive_grade  = minmax(0.30×tackles_won + 0.25×interceptions
                              + 0.25×clearances + 0.20×recoveries)   [DEF/MID/FWD]
    gk_grade         = minmax(saves_p90)                              [GK only]
    overall_grade:
      GK:  minmax(saves_p90)
      DEF: minmax(0.25×attack + 0.20×creativity + 0.55×defensive)
      MID: minmax(0.35×attack + 0.35×creativity + 0.30×defensive)
      FWD: minmax(0.50×attack + 0.35×creativity + 0.15×defensive)

    Parameters
    ----------
    player_season : pd.DataFrame
        Raw player_season.csv loaded as a DataFrame.
    events_df : pd.DataFrame, optional
        Full events.csv.  When supplied, adds defensive grades.
    lineups_df : pd.DataFrame, optional
        Full lineups.csv.  When supplied, adds GK-specific grades.

    Returns
    -------
    pd.DataFrame
        Columns: player, player_id, team, season, position, pos_group,
                 games, time, goals, assists,
                 npxg_p90, xa_p90, kp_p90, xg_chain_p90,
                 [tackles_won_p90, interceptions_p90, clearances_p90, recoveries_p90,]
                 [save_pct, saves_p90,]
                 attack_grade, creativity_grade, defensive_grade,
                 overall_grade
    """
    ps = player_season.copy()

    # ── Minimum playing time ──────────────────────────────────────────────────
    ps = ps[ps["time"] >= 90].copy()

    # ── Position group ────────────────────────────────────────────────────────
    ps["pos_group"] = ps["position"].apply(_pos_group)

    # ── Store p90 as a column BEFORE any merges ───────────────────────────────
    # Merges reset the DataFrame index, which would misalign a standalone p90
    # Series.  Storing it as "_p90" keeps it row-aligned through all joins.
    ps["_p90"] = ps["time"] / 90.0

    # ── Offensive per-90 metrics ──────────────────────────────────────────────
    ps["npxg_p90"]     = ps["npxg"]       / ps["_p90"]
    ps["xa_p90"]       = ps["xa"]         / ps["_p90"]
    ps["kp_p90"]       = ps["key_passes"] / ps["_p90"]
    ps["xg_chain_p90"] = ps["xg_chain"]   / ps["_p90"]

    # ── Defensive per-90 metrics (WhoScored events) ───────────────────────────
    has_defensive = events_df is not None and not events_df.empty
    if has_defensive:
        def_agg = _aggregate_defensive_events(events_df)

        # Join defensive action totals onto player_season on player+team+season.
        # ~85% of players match by exact name; the rest get NaN (median-imputed).
        ps = ps.merge(
            def_agg[["player", "team", "season",
                      "tackles_won", "interceptions", "clearances", "ball_recoveries"]],
            on=["player", "team", "season"],
            how="left",
        )

        # Scale totals to per-90 rates using the column (index-safe after merge)
        ps["tackles_won_p90"]   = ps["tackles_won"]     / ps["_p90"]
        ps["interceptions_p90"] = ps["interceptions"]   / ps["_p90"]
        ps["clearances_p90"]    = ps["clearances"]      / ps["_p90"]
        ps["recoveries_p90"]    = ps["ball_recoveries"] / ps["_p90"]
    else:
        for col in ["tackles_won_p90", "interceptions_p90",
                    "clearances_p90", "recoveries_p90"]:
            ps[col] = np.nan

    # ── Goalkeeper stats (ESPN lineups) ───────────────────────────────────────
    # Note: ESPN's shots_faced includes all attempts (on + off target), so
    # saves/shots_faced does not give a meaningful save percentage.
    # We use saves_p90 only.
    has_gk = lineups_df is not None and not lineups_df.empty
    if has_gk:
        gk_agg = _aggregate_gk_stats(lineups_df)
        ps = ps.merge(
            gk_agg[["player", "team", "season", "total_saves"]],
            on=["player", "team", "season"],
            how="left",
        )
        ps["saves_p90"] = ps["total_saves"] / ps["_p90"]
    else:
        ps["saves_p90"] = np.nan

    # ── Grade computation — scaled within season × position group ─────────────
    def_p90_cols = ["tackles_won_p90", "interceptions_p90",
                    "clearances_p90", "recoveries_p90"]
    off_p90_cols = ["npxg_p90", "xa_p90", "kp_p90", "xg_chain_p90"]
    gk_stat_cols = ["saves_p90"]

    grade_parts = []

    for (season, pos_group), grp in ps.groupby(["season", "pos_group"]):
        g = grp.copy()

        # Impute missing values with within-group median before scaling
        g = _impute_medians(g, off_p90_cols + def_p90_cols + gk_stat_cols)

        if pos_group == "GK":
            # ── Goalkeeper grade — saves per 90 only ─────────────────────────
            # ESPN shots_faced includes off-target attempts so save% is not
            # reliable; saves_p90 is used as the sole GK performance metric.
            g["defensive_grade"] = _minmax_scale(g["saves_p90"])
            g["attack_grade"]    = pd.Series(np.nan, index=g.index)
            g["creativity_grade"]= pd.Series(np.nan, index=g.index)
            g["overall_grade"]   = g["defensive_grade"]

        else:
            # ── Outfield: attack & creativity grades ─────────────────────────
            g["attack_grade"]  = _minmax_scale(g["npxg_p90"])

            creativity_raw     = 0.6 * g["xa_p90"] + 0.4 * g["kp_p90"]
            g["creativity_grade"] = _minmax_scale(creativity_raw)

            # ── Outfield: defensive grade ─────────────────────────────────────
            # Weights are the same across positions; the scaling within
            # position group already accounts for the fact that defenders
            # accumulate more clearances than forwards.
            def_raw = (
                0.30 * _minmax_scale(g["tackles_won_p90"])
                + 0.25 * _minmax_scale(g["interceptions_p90"])
                + 0.25 * _minmax_scale(g["clearances_p90"])
                + 0.20 * _minmax_scale(g["recoveries_p90"])
            )
            g["defensive_grade"] = def_raw  # weighted sum of [1,10] inputs → [1,10]

            # ── Overall grade — position-adjusted weights ─────────────────────
            weights = {
                "DEF": (0.25, 0.20, 0.55),
                "MID": (0.35, 0.35, 0.30),
                "FWD": (0.50, 0.35, 0.15),
            }.get(pos_group, (0.35, 0.35, 0.30))

            g["overall_grade"] = (
                weights[0] * g["attack_grade"]
                + weights[1] * g["creativity_grade"]
                + weights[2] * g["defensive_grade"]
            )

        grade_parts.append(g)

    result = pd.concat(grade_parts, ignore_index=True)

    # ── Output columns ────────────────────────────────────────────────────────
    out_cols = [
        "player", "player_id", "team", "season", "position", "pos_group",
        "games", "time", "goals", "assists",
        "npxg_p90", "xa_p90", "kp_p90", "xg_chain_p90",
        "tackles_won_p90", "interceptions_p90", "clearances_p90", "recoveries_p90",
        "saves_p90",
        "attack_grade", "creativity_grade", "defensive_grade", "overall_grade",
    ]
    out_cols = [c for c in out_cols if c in result.columns]
    return result[out_cols].reset_index(drop=True)
