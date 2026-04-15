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

    rows = []

    for _, match in ms.iterrows():
        # Build a perspective dict for each side
        sides = [
            dict(
                side="home",
                team=match["home_team"],
                opponent=match["away_team"],
                is_home=True,
                goals=match["home_goals"],
                goals_conceded=match["away_goals"],
                xg=match["home_xg"],
                xga=match["away_xg"],
                shots=match.get("home_total_shots", np.nan),
                shots_on_target=match.get("home_shots_on_target", np.nan),
                possession_pct=match.get("home_possession_pct", np.nan),
                pass_pct=match.get("home_pass_pct", np.nan),
                tackle_pct=match.get("home_tackle_pct", np.nan),
                interceptions=match.get("home_interceptions", np.nan),
                saves=match.get("home_saves", np.nan),
                fouls=match.get("home_fouls_committed", np.nan),
                yellow_cards=match.get("home_yellow_cards", np.nan),
                red_cards=match.get("home_red_cards", np.nan),
            ),
            dict(
                side="away",
                team=match["away_team"],
                opponent=match["home_team"],
                is_home=False,
                goals=match["away_goals"],
                goals_conceded=match["home_goals"],
                xg=match["away_xg"],
                xga=match["home_xg"],
                shots=match.get("away_total_shots", np.nan),
                shots_on_target=match.get("away_shots_on_target", np.nan),
                possession_pct=match.get("away_possession_pct", np.nan),
                pass_pct=match.get("away_pass_pct", np.nan),
                tackle_pct=match.get("away_tackle_pct", np.nan),
                interceptions=match.get("away_interceptions", np.nan),
                saves=match.get("away_saves", np.nan),
                fouls=match.get("away_fouls_committed", np.nan),
                yellow_cards=match.get("away_yellow_cards", np.nan),
                red_cards=match.get("away_red_cards", np.nan),
            ),
        ]

        for s in sides:
            g = s["goals"]
            gc = s["goals_conceded"]

            # Result / points from this team's perspective
            if pd.notna(g) and pd.notna(gc):
                if g > gc:
                    result, points = "W", 3
                elif g == gc:
                    result, points = "D", 1
                else:
                    result, points = "L", 0
            else:
                result, points = np.nan, np.nan

            # Derived shooting quality metrics
            shots = s["shots"]
            sot = s["shots_on_target"]
            xg = s["xg"]

            sot_pct = (sot / shots * 100) if (pd.notna(shots) and pd.notna(sot) and shots > 0) else np.nan
            xg_per_shot = (xg / shots) if (pd.notna(xg) and pd.notna(shots) and shots > 0) else np.nan

            rows.append(
                {
                    "season": match["season"],
                    "match_date": match["match_date"],
                    "team": s["team"],
                    "opponent": s["opponent"],
                    "is_home": s["is_home"],
                    "goals": g,
                    "goals_conceded": gc,
                    "xg": xg,
                    "xga": s["xga"],
                    "shots": shots,
                    "shots_on_target": sot,
                    "shots_on_target_pct": sot_pct,
                    "xg_per_shot": xg_per_shot,
                    "possession_pct": s["possession_pct"],
                    "pass_pct": s["pass_pct"],
                    "tackle_pct": s["tackle_pct"],
                    "interceptions": s["interceptions"],
                    "saves": s["saves"],
                    "fouls": s["fouls"],
                    "yellow_cards": s["yellow_cards"],
                    "red_cards": s["red_cards"],
                    "result": result,
                    "points": points,
                }
            )

    flat = pd.DataFrame(rows)
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

        # Re-scale composites back to [1,10] so each sub-grade is self-consistent
        g["attack_grade"]  = _minmax_scale(attack_raw)
        g["defense_grade"] = _minmax_scale(defense_raw)
        g["style_grade"]   = _minmax_scale(style_raw)

        overall_raw = (
            0.40 * g["attack_grade"]
            + 0.40 * g["defense_grade"]
            + 0.20 * g["style_grade"]
        )
        g["overall_grade"] = _minmax_scale(overall_raw)

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

        g["roll_attack_grade"]  = _minmax_scale(attack_raw)
        g["roll_defense_grade"] = _minmax_scale(defense_raw)
        g["roll_style_grade"]   = _minmax_scale(style_raw)

        overall_raw = (
            0.40 * g["roll_attack_grade"]
            + 0.40 * g["roll_defense_grade"]
            + 0.20 * g["roll_style_grade"]
        )
        g["roll_overall_grade"] = _minmax_scale(overall_raw)

        grade_parts.append(g)

    return pd.concat(grade_parts).sort_values(["season", "match_date", "team"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4.  Player season grades
# ---------------------------------------------------------------------------

def compute_player_grades(player_season: pd.DataFrame) -> pd.DataFrame:
    """Compute 1-10 grades for players, one row per player × season.

    Only players with at least 90 minutes played are included (sufficient
    sample to compute meaningful per-90 metrics).

    Per-90 metrics
    --------------
    npxg_p90     = npxg / (time / 90)   — non-penalty xG per 90 min
    xa_p90       = xa   / (time / 90)   — expected assists per 90 min
    kp_p90       = key_passes / (time / 90)
    xg_chain_p90 = xg_chain  / (time / 90)

    Grade formulas (scaled within season)
    --------------------------------------
    attack_grade     = minmax(npxg_p90)
    creativity_score = 0.6×xa_p90 + 0.4×kp_p90   (raw composite)
    creativity_grade = minmax(creativity_score)
    chain_scaled     = minmax(xg_chain_p90)
    overall_grade    = minmax(0.5×attack_grade + 0.3×creativity_grade + 0.2×chain_scaled)

    Parameters
    ----------
    player_season : pd.DataFrame
        Raw player_season.csv loaded as a DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: player, player_id, team, season, position, games, time,
                 goals, assists, npxg_p90, xa_p90, kp_p90, xg_chain_p90,
                 attack_grade, creativity_grade, overall_grade
    """
    ps = player_season.copy()

    # Minimum playing time filter
    ps = ps[ps["time"] >= 90].copy()

    # Minutes per 90 denominator
    p90 = ps["time"] / 90.0

    # Compute per-90 metrics
    ps["npxg_p90"]     = ps["npxg"]      / p90
    ps["xa_p90"]       = ps["xa"]        / p90
    ps["kp_p90"]       = ps["key_passes"] / p90
    ps["xg_chain_p90"] = ps["xg_chain"]  / p90

    per90_cols = ["npxg_p90", "xa_p90", "kp_p90", "xg_chain_p90"]

    # Median-impute per-90 metrics within season before scaling
    grade_parts = []

    for season, grp in ps.groupby("season"):
        g = grp.copy()
        g = _impute_medians(g, per90_cols)

        # Scale individual metrics to [1,10]
        g["attack_grade"]     = _minmax_scale(g["npxg_p90"])

        creativity_raw        = 0.6 * g["xa_p90"] + 0.4 * g["kp_p90"]
        g["creativity_grade"] = _minmax_scale(creativity_raw)

        chain_scaled          = _minmax_scale(g["xg_chain_p90"])

        # Overall: re-scale the weighted combination back to [1,10]
        overall_raw           = (
            0.5 * g["attack_grade"]
            + 0.3 * g["creativity_grade"]
            + 0.2 * chain_scaled
        )
        g["overall_grade"]    = _minmax_scale(overall_raw)

        grade_parts.append(g)

    result = pd.concat(grade_parts, ignore_index=True)

    # Return only the specified output columns
    out_cols = [
        "player", "player_id", "team", "season", "position",
        "games", "time", "goals", "assists",
        "npxg_p90", "xa_p90", "kp_p90", "xg_chain_p90",
        "attack_grade", "creativity_grade", "overall_grade",
    ]
    # Keep only columns that actually exist (graceful degradation)
    out_cols = [c for c in out_cols if c in result.columns]
    return result[out_cols].reset_index(drop=True)
