"""
sanitize.py
-----------
Normalization utilities shared across collectors and the build pipeline.

Handles:
  - Team name canonicalization across Understat / ESPN / WhoScored
  - Coordinate normalization to the WhoScored [0, 100] standard
  - Date normalization to UTC-aware timestamps

Canonical team names use the short form (e.g. "Man City", "Wolves").
All three sources resolve to these names via TEAM_NAME_MAP.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Team name mapping
# ---------------------------------------------------------------------------

# Short canonical names used in all processed output.
# Add new entries here whenever a new team appears in any source.
TEAM_NAME_MAP: dict[str, str] = {
    # ---- Canonical names (pass-through) ----
    "Arsenal":           "Arsenal",
    "Aston Villa":       "Aston Villa",
    "Bournemouth":       "Bournemouth",
    "Brentford":         "Brentford",
    "Brighton":          "Brighton",
    "Burnley":           "Burnley",
    "Chelsea":           "Chelsea",
    "Crystal Palace":    "Crystal Palace",
    "Everton":           "Everton",
    "Fulham":            "Fulham",
    "Leeds":             "Leeds",
    "Liverpool":         "Liverpool",
    "Man City":          "Man City",
    "Man Utd":           "Man Utd",
    "Newcastle":         "Newcastle",
    "Nottingham Forest": "Nottingham Forest",
    "Sunderland":        "Sunderland",
    "Tottenham":         "Tottenham",
    "West Ham":          "West Ham",
    "Wolves":            "Wolves",
    # ---- Understat variants ----
    "Manchester City":         "Man City",
    "Manchester United":       "Man Utd",
    "Newcastle United":        "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    # ---- ESPN variants ----
    "AFC Bournemouth":         "Bournemouth",
    "Brighton & Hove Albion":  "Brighton",
    "Leeds United":            "Leeds",
    "Tottenham Hotspur":       "Tottenham",
    "West Ham United":         "West Ham",
    # ESPN uses long names for Man City/Utd too — already covered above
}


def canonicalize_team(name: str) -> str:
    """Return the canonical short name for any source spelling.

    Handles comma-separated multi-team values produced by Understat for
    players who transferred mid-season (e.g. "Bournemouth,Manchester City").
    Each part is canonicalized individually and the string is re-joined.

    Falls back to the original (stripped) name if not in the map,
    so new promoted/relegated teams degrade gracefully rather than erroring.
    """
    if pd.isna(name):
        return name
    name = str(name).strip()

    # Multi-team (transfer): canonicalize each part separately
    if "," in name:
        parts = [_lookup_team(p.strip()) for p in name.split(",")]
        return ",".join(parts)

    return _lookup_team(name)


def _lookup_team(name: str) -> str:
    """Single-team lookup with unknown-name warning."""
    canonical = TEAM_NAME_MAP.get(name)
    if canonical is None:
        if name not in _unknown_teams_warned:
            print(f"  ⚠ sanitize: unknown team '{name}' — using as-is. "
                  f"Add to TEAM_NAME_MAP if needed.")
            _unknown_teams_warned.add(name)
        return name
    return canonical


_unknown_teams_warned: set[str] = set()   # module-level cache to avoid repeat warnings


def canonicalize_teams(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Apply canonicalize_team to one or more columns in a DataFrame (in-place).

    Example
    -------
    canonicalize_teams(df, "home_team", "away_team")
    """
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(canonicalize_team)
    return df


# ---------------------------------------------------------------------------
# Coordinate normalization
# ---------------------------------------------------------------------------
#
# Standard: WhoScored [0, 100] coordinate system
#   x = 0   → defending goal line
#   x = 100 → attacking goal line
#   y = 0   → top touchline (left when attacking left→right)
#   y = 100 → bottom touchline
#
# Understat uses the same orientation but normalised to [0, 1] fractions,
# so the conversion is simply × 100. No axis flip required — verified by
# comparing shot x distributions: both sources cluster near 85–90 for
# shots on target (i.e. near the attacking end).
# ---------------------------------------------------------------------------

def normalize_coords(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    source: str = "understat",
) -> pd.DataFrame:
    """
    Normalize shot / event coordinates to [0, 100].

    Parameters
    ----------
    df     : DataFrame containing coordinate columns.
    x_col  : Name of the x-coordinate column.
    y_col  : Name of the y-coordinate column.
    source : 'understat' (scales × 100) or 'whoscored' (no-op, already [0,100]).
    """
    df = df.copy()
    if source == "understat":
        df[x_col] = (df[x_col] * 100).round(2)
        df[y_col] = (df[y_col] * 100).round(2)
    # 'whoscored' is already in [0, 100] — intentional no-op
    return df


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

def normalize_date(series: pd.Series) -> pd.Series:
    """Parse a date Series to UTC-aware timestamps, floored to the second.

    Works on both tz-naive strings (Understat) and tz-aware strings (ESPN/WhoScored).
    """
    return pd.to_datetime(series, utc=True, errors="coerce").dt.floor("s")
