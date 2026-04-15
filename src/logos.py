"""
logos.py
--------
Club crest utilities for the Premier League Analytics dashboard.

Crests are sourced from the ESPN CDN and cached locally in assets/logos/.
Run `python fetch_logos.py` once to download all crests to disk.

Usage
-----
from src.logos import logo_path, logo_url, TEAM_LOGO_IDS

# Get a local file path (falls back to ESPN URL if not cached):
path = logo_path("Arsenal")      # "assets/logos/Arsenal.png" (if downloaded)

# Get the ESPN CDN URL directly:
url = logo_url("Arsenal")        # "https://a.espncdn.com/i/teamlogos/soccer/500/359.png"

# Use in matplotlib (with mpimg / OffsetImage):
from src.logos import add_logo_to_ax
add_logo_to_ax(ax, "Arsenal", xy=(0.05, 0.9), size=0.08)
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

# ---------------------------------------------------------------------------
# ESPN team ID mapping  →  canonical_name : espn_id
# ---------------------------------------------------------------------------
# Logo URL template: https://a.espncdn.com/i/teamlogos/soccer/500/{id}.png

TEAM_LOGO_IDS: dict[str, int] = {
    "Arsenal":           359,
    "Aston Villa":       362,
    "Bournemouth":       349,
    "Brentford":         337,
    "Brighton":          331,
    "Burnley":           379,
    "Chelsea":           363,
    "Crystal Palace":    384,
    "Everton":           368,
    "Fulham":            370,
    "Ipswich":           373,
    "Leeds":             357,
    "Leicester":         375,
    "Liverpool":         364,
    "Luton":             342,
    "Man City":          382,
    "Man Utd":           360,
    "Newcastle":         361,
    "Norwich":           387,
    "Nottingham Forest": 393,
    "Sheffield Utd":     336,
    "Southampton":       374,   # Southampton ESPN id
    "Sunderland":        366,
    "Tottenham":         367,
    "Watford":           376,
    "West Brom":         383,
    "West Ham":          371,
    "Wolves":            380,
}

# Root of the project (two levels up from this file: src/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOGOS_DIR    = _PROJECT_ROOT / "assets" / "logos"

ESPN_CDN = "https://a.espncdn.com/i/teamlogos/soccer/500/{id}.png"


def logo_url(team: str) -> str | None:
    """Return the ESPN CDN URL for a team's crest, or None if unknown."""
    tid = TEAM_LOGO_IDS.get(team)
    return ESPN_CDN.format(id=tid) if tid else None


def logo_path(team: str) -> str | None:
    """Return the local cached crest path if it exists, else None.

    Download all crests first with `python fetch_logos.py`.
    """
    if team not in TEAM_LOGO_IDS:
        return None
    path = _LOGOS_DIR / f"{team}.png"
    return str(path) if path.exists() else None


def load_logo(team: str) -> np.ndarray | None:
    """Load a team crest as a numpy RGBA array.

    Tries the local cache first; returns None if neither is available.
    Matplotlib's imread is used so the result can be passed directly to
    OffsetImage for embedding in charts.
    """
    path = logo_path(team)
    if path:
        try:
            return mpimg.imread(path)
        except Exception:
            pass
    return None


def add_logo_to_ax(
    ax: plt.Axes,
    team: str,
    xy: tuple[float, float] = (0.5, 0.5),
    size: float = 0.08,
    alpha: float = 0.85,
    coords: str = "axes fraction",
    zorder: int = 10,
) -> None:
    """Embed a team crest into a matplotlib Axes.

    Parameters
    ----------
    ax     : Target Axes object.
    team   : Canonical team name (must be in TEAM_LOGO_IDS).
    xy     : (x, y) position in `coords` space (default: axes fraction).
    size   : Zoom factor passed to OffsetImage — larger = bigger crest.
    alpha  : Transparency (0=invisible, 1=opaque).
    coords : Coordinate system for `xy` (e.g. "axes fraction", "data").
    zorder : Drawing order (higher = on top).
    """
    img = load_logo(team)
    if img is None:
        return

    imagebox = OffsetImage(img, zoom=size, alpha=alpha)
    ab = AnnotationBbox(
        imagebox, xy,
        xycoords=coords,
        frameon=False,
        zorder=zorder,
    )
    ax.add_artist(ab)


def logo_html(team: str, size: int = 24) -> str:
    """Return an HTML <img> tag for a team crest, for use in st.markdown().

    Uses the ESPN CDN URL directly (no local file required).
    Returns an empty string if the team is unknown.

    Example
    -------
    st.markdown(logo_html("Arsenal") + " Arsenal", unsafe_allow_html=True)
    """
    url = logo_url(team)
    if not url:
        return ""
    return (
        f'<img src="{url}" width="{size}" height="{size}" '
        f'style="vertical-align:middle; margin-right:6px;" />'
    )
