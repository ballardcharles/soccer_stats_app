"""
predictor.py — Poisson regression match predictor for EPL fixtures.

Overview
--------
The model assigns each team an attack strength and a defense strength relative
to the league average.  Given a fixture, expected goals for each side are
computed as:

    lambda_home = league_avg * attack(home) * defense(away) * home_advantage
    lambda_away = league_avg * attack(away) * defense(home)

Goals for each team are treated as independent Poisson random variables, so the
joint probability of any scoreline (i, j) is:

    P(home=i, away=j) = Poisson(i; lambda_home) * Poisson(j; lambda_away)

Summing over the 7×7 scoreline grid (0–6 goals per side) yields W/D/L
probabilities and the most likely individual scorelines.

Ratings use the most recent n_recent matches per team and are shrunk toward
the league average (1.0) via a simple Bayesian prior to prevent extreme
estimates from teams with few recent games.

References
----------
- Maher (1982) "Modelling association football scores"
- Dixon & Coles (1997) "Modelling association football scores and inefficiencies
  in the football betting market"
"""

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

SHRINK_FACTOR = 5        # Bayesian shrinkage toward league average (1.0)
MAX_GOALS     = 6        # Poisson grid upper bound (inclusive)
LAMBDA_MIN    = 0.3      # lower clamp for expected goals
LAMBDA_MAX    = 4.5      # upper clamp for expected goals
DEFAULT_RATING = 0.85    # prior for promoted / unknown teams (slightly below avg)


# ---------------------------------------------------------------------------
# get_upcoming_fixtures
# ---------------------------------------------------------------------------

def get_upcoming_fixtures(crossref: pd.DataFrame) -> pd.DataFrame:
    """Return unplayed fixtures from *crossref*, sorted by date ascending.

    A fixture is considered unplayed when ``home_goals`` is NaN (Understat
    has not yet recorded a result).

    Parameters
    ----------
    crossref:
        DataFrame loaded from ``data/processed/match_crossref.csv``.  Must
        contain columns ``match_date``, ``home_team``, ``away_team``, and
        ``home_goals``.

    Returns
    -------
    pd.DataFrame
        Filtered copy with ``match_date`` parsed to ``datetime64`` and a new
        ``fixture_label`` column formatted as ``"19 Apr  Arsenal v Man City"``.
    """
    if crossref.empty:
        return crossref.copy()

    df = crossref.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    upcoming = df[df["home_goals"].isna()].copy()
    upcoming = upcoming.sort_values("match_date").reset_index(drop=True)

    # Build a human-readable label: "19 Apr  Arsenal v Man City"
    def _label(row: pd.Series) -> str:
        date_str = row["match_date"].strftime("%-d %b")   # e.g. "19 Apr"
        return f"{date_str}  {row['home_team']} v {row['away_team']}"

    upcoming["fixture_label"] = upcoming.apply(_label, axis=1)
    return upcoming


# ---------------------------------------------------------------------------
# build_poisson_model
# ---------------------------------------------------------------------------

def build_poisson_model(crossref: pd.DataFrame, n_recent: int = 10) -> dict:
    """Build Poisson attack/defence strength ratings from recent match data.

    Uses expected goals (xG / xGA) rather than actual goals because xG is a
    more stable estimator of true team quality over small sample sizes.

    Algorithm
    ---------
    1. Filter to completed matches (``home_goals`` not NaN).
    2. Compute the league average xG per team per match as the baseline.
    3. For each team, take their ``n_recent`` most recent appearances (home or
       away) and compute raw attack and defence ratings relative to the league
       average.
    4. Shrink each raw rating toward 1.0 (league average) using a Bayesian
       prior weighted by ``SHRINK_FACTOR``.
    5. Compute the home-field advantage as the ratio of total home xG to total
       away xG across all completed matches.

    Parameters
    ----------
    crossref:
        DataFrame from ``data/processed/match_crossref.csv``.
    n_recent:
        Number of most-recent matches per team used to estimate ratings.
        Older matches are discarded to reflect current form.

    Returns
    -------
    dict with keys:

    - ``"team_attack"``  : dict[str, float] — attack multiplier (1.0 = avg)
    - ``"team_defense"`` : dict[str, float] — defence multiplier (1.0 = avg)
    - ``"league_avg"``   : float — mean xG per team per match
    - ``"home_advantage"``: float — home scoring boost factor (~1.1–1.2)
    """
    if crossref.empty:
        return _empty_model()

    df = crossref.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    completed = df[df["home_goals"].notna()].copy()

    if completed.empty:
        return _empty_model()

    # ------------------------------------------------------------------
    # Step 2 — league baseline
    # ------------------------------------------------------------------
    total_home_xg = completed["home_xg"].sum()
    total_away_xg = completed["away_xg"].sum()
    total_xg      = total_home_xg + total_away_xg
    n_matches      = len(completed)

    if n_matches == 0 or total_xg == 0:
        return _empty_model()

    # Each match contributes xG for two teams, so per-team-per-match average
    # equals total_xg / (2 * n_matches) — but because each Poisson lambda
    # already represents one side's expected goals, we use the per-match
    # average across all "team slots":
    #   league_avg = total_xg / (2 * n_matches)
    league_avg: float = total_xg / (2.0 * n_matches)

    # ------------------------------------------------------------------
    # Step 5 — home advantage (computed on full dataset for stability)
    # ------------------------------------------------------------------
    home_advantage: float = (
        total_home_xg / total_away_xg if total_away_xg > 0 else 1.0
    )

    # ------------------------------------------------------------------
    # Step 3 — collect each team's recent matches
    # ------------------------------------------------------------------
    # Build a long-form view: for every completed match, create a home-team
    # record (xg_scored = home_xg, xg_conceded = away_xg) and an away-team
    # record (xg_scored = away_xg, xg_conceded = home_xg).
    home_rows = completed[["match_date", "home_team", "home_xg", "away_xg"]].copy()
    home_rows.columns = ["match_date", "team", "xg_scored", "xg_conceded"]

    away_rows = completed[["match_date", "away_team", "away_xg", "home_xg"]].copy()
    away_rows.columns = ["match_date", "team", "xg_scored", "xg_conceded"]

    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long = long.sort_values("match_date", ascending=False)

    # ------------------------------------------------------------------
    # Step 3–4 — per-team ratings with shrinkage
    # ------------------------------------------------------------------
    team_attack:  dict[str, float] = {}
    team_defense: dict[str, float] = {}

    all_teams = long["team"].unique()
    for team in all_teams:
        recent = long[long["team"] == team].head(n_recent)
        n = len(recent)

        if n == 0:
            # No data — use default prior
            team_attack[team]  = DEFAULT_RATING
            team_defense[team] = DEFAULT_RATING
            continue

        avg_scored    = recent["xg_scored"].mean()
        avg_conceded  = recent["xg_conceded"].mean()

        raw_attack  = avg_scored   / league_avg if league_avg > 0 else 1.0
        raw_defense = avg_conceded / league_avg if league_avg > 0 else 1.0

        # Bayesian shrinkage toward 1.0
        team_attack[team]  = _shrink(raw_attack,  n, SHRINK_FACTOR)
        team_defense[team] = _shrink(raw_defense, n, SHRINK_FACTOR)

    return {
        "team_attack":     team_attack,
        "team_defense":    team_defense,
        "league_avg":      league_avg,
        "home_advantage":  home_advantage,
    }


# ---------------------------------------------------------------------------
# predict_fixture
# ---------------------------------------------------------------------------

def predict_fixture(home_team: str, away_team: str, model: dict) -> dict:
    """Predict match outcome probabilities using a Poisson goals model.

    The joint probability of any scoreline (i, j) is modelled as the product
    of two independent Poisson distributions:

        P(home=i) = e^{-λ_h} * λ_h^i / i!
        P(away=j) = e^{-λ_a} * λ_a^j / j!

    Outcome probabilities are obtained by summing P(i,j) over the 7×7 grid
    (goals 0 through ``MAX_GOALS``) and then normalising to 1.

    Parameters
    ----------
    home_team, away_team:
        Team names exactly as they appear in the model.  Unknown teams receive
        a default attack and defence rating of ``DEFAULT_RATING`` (0.85).
    model:
        Dict returned by :func:`build_poisson_model`.

    Returns
    -------
    dict with keys:

    - ``"home_win_prob"``  : float
    - ``"draw_prob"``      : float
    - ``"away_win_prob"``  : float
    - ``"exp_home_goals"`` : float   (λ_home, before clamping)
    - ``"exp_away_goals"`` : float   (λ_away, before clamping)
    - ``"top_scorelines"`` : list[tuple[str, float]]  — top 6 by probability
    - ``"lambda_home"``    : float   (clamped λ used in calculations)
    - ``"lambda_away"``    : float   (clamped λ used in calculations)
    """
    atk  = model.get("team_attack",  {})
    dfn  = model.get("team_defense", {})
    avg  = model.get("league_avg",   1.2)
    hadv = model.get("home_advantage", 1.1)

    home_atk  = atk.get(home_team,  DEFAULT_RATING)
    home_def  = dfn.get(home_team,  DEFAULT_RATING)
    away_atk  = atk.get(away_team,  DEFAULT_RATING)
    away_def  = dfn.get(away_team,  DEFAULT_RATING)

    # Expected goals before clamping (for reporting)
    exp_home = avg * home_atk * away_def * hadv
    exp_away = avg * away_atk * home_def

    # Clamped lambdas used in Poisson PMF
    lam_h = max(LAMBDA_MIN, min(LAMBDA_MAX, exp_home))
    lam_a = max(LAMBDA_MIN, min(LAMBDA_MAX, exp_away))

    # ------------------------------------------------------------------
    # Build 7×7 Poisson PMF grid
    # ------------------------------------------------------------------
    home_pmf = [_poisson_pmf(lam_h, k) for k in range(MAX_GOALS + 1)]
    away_pmf = [_poisson_pmf(lam_a, k) for k in range(MAX_GOALS + 1)]

    home_win = 0.0
    draw     = 0.0
    away_win = 0.0
    scorelines: list[tuple[str, float]] = []

    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = home_pmf[i] * away_pmf[j]
            scorelines.append((f"{i}-{j}", p))
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p

    # Normalise W/D/L so they sum exactly to 1.0
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw     /= total
        away_win /= total

    # Top 6 most likely scorelines
    scorelines.sort(key=lambda x: x[1], reverse=True)
    top_scorelines = scorelines[:6]

    return {
        "home_win_prob":  home_win,
        "draw_prob":      draw,
        "away_win_prob":  away_win,
        "exp_home_goals": exp_home,
        "exp_away_goals": exp_away,
        "top_scorelines": top_scorelines,
        "lambda_home":    lam_h,
        "lambda_away":    lam_a,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _poisson_pmf(lam: float, k: int) -> float:
    """Compute the Poisson probability mass function P(X=k; lambda).

    Uses the formula directly (no scipy dependency):
        P(X=k) = e^{-lam} * lam^k / k!
    """
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _shrink(raw: float, n: int, shrink: float) -> float:
    """Bayesian shrinkage of *raw* toward 1.0 (league average).

    With few observations the posterior is pulled strongly toward the prior
    (1.0 = league average).  As n grows the raw estimate dominates.

        posterior = (n * raw + shrink * 1.0) / (n + shrink)
    """
    return (n * raw + shrink * 1.0) / (n + shrink)


def _empty_model() -> dict:
    """Return a neutral model used when no completed match data is available."""
    return {
        "team_attack":    {},
        "team_defense":   {},
        "league_avg":     1.2,   # typical EPL xG per team per match
        "home_advantage": 1.1,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv  = os.path.join(base, "data", "processed", "match_crossref.csv")
    df   = pd.read_csv(csv)

    model    = build_poisson_model(df, n_recent=10)
    upcoming = get_upcoming_fixtures(df)

    print(f"League avg xG/team/match : {model['league_avg']:.3f}")
    print(f"Home advantage factor    : {model['home_advantage']:.3f}")
    print(f"Teams rated              : {len(model['team_attack'])}")
    print()
    print("Next 5 fixtures:")
    for _, row in upcoming.head(5).iterrows():
        result = predict_fixture(row["home_team"], row["away_team"], model)
        print(
            f"  {row['fixture_label']:<35}  "
            f"H {result['home_win_prob']:.0%}  "
            f"D {result['draw_prob']:.0%}  "
            f"A {result['away_win_prob']:.0%}  "
            f"({result['lambda_home']:.2f} – {result['lambda_away']:.2f})"
        )
