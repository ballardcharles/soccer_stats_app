"""
dashboard.py
------------
Premier League Analytics Dashboard
Data: Understat · ESPN · WhoScored  →  data/processed/

Run with:
    streamlit run dashboard.py

Views available:
    📋 League Table   — standings, form, goal stats and next fixture
    🎯 Shot Maps      — shot locations and xG per team / player / match
    🔥 Event Heatmaps — WhoScored event density maps by type and period
    📊 Match Analysis — per-match scorecard, xG bar, forecast, shot map
    👤 Player Stats   — season stats table with xG and xA scatter plots
    📈 xG Analysis    — team + player xG performance, over/underperformance, hover tooltips
    🏅 Team Grades    — 1-10 attack / defense / style grades, rolling form
    ⭐ Player Grades  — per-90 offensive + defensive grades by position
    🔮 Match Predictor — Poisson model W/D/L probabilities for upcoming fixtures
    ⚡ Shot Quality   — zone quality hexbin maps and team shot-quality rankings
    ⚔️ Head-to-Head   — radar comparison, H2H record, and key stats table
"""

import io
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.lines import Line2D
from mplsoccer import Pitch, VerticalPitch

# Local modules
sys.path.insert(0, "src")
from logos import logo_html, logo_path, add_logo_to_ax, load_logo, TEAM_LOGO_IDS
from scoring import (
    flatten_match_summary,
    compute_season_grades,
    compute_rolling_grades,
    compute_player_grades,
)
from predictor import (
    build_poisson_model,
    get_upcoming_fixtures,
    predict_fixture,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════════════════════
# set_page_config must be the first Streamlit call in the script.
# layout="wide" uses the full browser width instead of the default narrow column.

st.set_page_config(
    page_title="Premier League Analytics",
    layout="wide",
    page_icon="⚽",
)

# Responsive stylesheet — dark theme on all screen sizes, mobile-optimised
# layout on phones (≤ 768 px), full wide layout on tablets and desktop.
# unsafe_allow_html=True is required to pass raw HTML/CSS into Streamlit.
st.markdown("""
<style>

/* ── Base styles (all screen sizes) ───────────────────────────────────────── */
.main { background-color: #0e1117; }
h1, h2, h3 { color: #e8e8e8; }
.stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 5px; }

/* ── Mobile  (≤ 768 px) ────────────────────────────────────────────────────
   Goals:
     • Remove excess horizontal padding so charts fill the screen
     • Stack every multi-column row vertically
     • Scale typography down slightly
     • Make metric cards, dataframes, and tabs usable on a small screen
     • Keep touch targets large enough to tap comfortably
────────────────────────────────────────────────────────────────────────── */
@media screen and (max-width: 768px) {

    /* Give the main content breathing room but not so much it wastes space */
    .block-container {
        padding-left:  0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top:   3rem   !important;   /* clear the hamburger button */
        max-width:     100vw  !important;
    }

    /* Stack every column row vertically — covers all st.columns() calls */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stColumn"] {
        width:     100% !important;
        flex:      1 1 100% !important;
        min-width: 100% !important;
    }

    /* Typography: slightly smaller but still readable */
    h1 { font-size: 1.4rem !important; line-height: 1.3 !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1.0rem  !important; }
    p  { font-size: 0.9rem  !important; }
    small,
    [data-testid="stCaptionContainer"] p {
        font-size: 0.78rem !important;
    }

    /* Metric cards: tighter padding, smaller label, legible value */
    [data-testid="metric-container"] {
        padding: 0.4rem 0.6rem !important;
    }
    [data-testid="metric-container"] label {
        font-size: 0.72rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
    }

    /* DataFrames: scroll horizontally rather than overflowing */
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrameResizable"] > div {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }

    /* Tabs: scroll horizontally if labels don't fit */
    [data-testid="stTabs"] [role="tablist"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        flex-wrap: nowrap !important;
        scrollbar-width: none !important;
    }
    [data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {
        display: none !important;
    }
    [data-testid="stTabs"] [role="tab"] {
        white-space: nowrap !important;
        font-size: 0.8rem !important;
        padding: 0.3rem 0.6rem !important;
    }

    /* Horizontal radio buttons (e.g. Heatmap / Arrow Map toggle): wrap */
    [data-testid="stRadio"] > div[role="radiogroup"] {
        flex-wrap: wrap !important;
        gap: 0.4rem !important;
    }

    /* Inputs: full width and easier to tap */
    [data-testid="stSelectbox"],
    [data-testid="stMultiSelect"],
    [data-testid="stSlider"] {
        width: 100% !important;
    }

    /* Buttons: easier tap target */
    [data-testid="stButton"] button {
        min-height: 2.25rem !important;
        font-size: 0.85rem !important;
    }

    /* Charts: Plotly fills full column width */
    .js-plotly-plot,
    .js-plotly-plot .plotly {
        width: 100% !important;
    }

    /* Dividers: tighter spacing */
    hr { margin: 0.5rem 0 !important; }

    /* Top decoration bar Streamlit adds — not useful on mobile */
    [data-testid="stDecoration"] { display: none !important; }
}

/* ── Tablet  (769 px – 1024 px) ────────────────────────────────────────────
   Mostly fine at this size — just trim padding so charts aren't cramped.
   Columns do NOT stack; the wide layout is preserved.
────────────────────────────────────────────────────────────────────────── */
@media screen and (min-width: 769px) and (max-width: 1024px) {
    .block-container {
        padding-left:  1.5rem !important;
        padding-right: 1.5rem !important;
    }
}

</style>
""", unsafe_allow_html=True)

# Shared colour constants used across all charts so every view stays consistent.
PURPLE      = "#37003c"   # Premier League deep purple (bars, accents)
GREEN       = "#00ff85"   # Premier League neon green (highlights, goals)
PITCH_GREEN = "#1a472a"   # Dark grass green used as the pitch background

# Paths — DB is preferred; CSVs are the fallback for local dev without a built DB.
DB_PATH   = "soccer_stats.db"
PROCESSED = "data/processed"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
# Prefers soccer_stats.db (SQLite) for fast, memory-efficient loading.
# Falls back to individual CSVs in data/processed/ for local dev convenience.
# @st.cache_data ensures the data is read from disk only once per session.

@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    """Load all data tables except events into a single dictionary of DataFrames.

    Events are intentionally excluded here — at 1 GB+ in RAM they would crash
    Streamlit Cloud's 1 GB memory limit before the app renders a single chart.
    Use load_events(season) instead wherever events are needed.

    Reads from soccer_stats.db (SQLite) if it exists, otherwise falls back
    to the individual CSVs in data/processed/.
    """
    import os, sqlite3

    def _parse_dates(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        for key in ("shots", "match_summary", "lineups", "match_crossref"):
            df = dfs.get(key, pd.DataFrame())
            if "match_date" in df.columns:
                df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        return dfs

    # ── Primary path: SQLite ──────────────────────────────────────────────────
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        def _tbl(name: str) -> pd.DataFrame:
            try:
                return pd.read_sql_query(f"SELECT * FROM {name}", conn)
            except Exception:
                return pd.DataFrame()

        dfs = dict(
            shots         = _tbl("shots"),
            events        = pd.DataFrame(),   # loaded lazily via load_events()
            match_summary = _tbl("match_summary"),
            match_crossref= _tbl("match_crossref"),
            player_season = _tbl("player_season"),
            lineups       = _tbl("lineups"),
            formations    = _tbl("formations"),
        )
        conn.close()
        return _parse_dates(dfs)

    # ── Fallback path: CSVs ───────────────────────────────────────────────────
    def safe_read(name: str) -> pd.DataFrame:
        try:
            return pd.read_csv(f"{PROCESSED}/{name}", low_memory=False)
        except FileNotFoundError:
            return pd.DataFrame()

    dfs = dict(
        shots         = safe_read("shots.csv"),
        events        = pd.DataFrame(),       # loaded lazily via load_events()
        match_summary = safe_read("match_summary.csv"),
        match_crossref= safe_read("match_crossref.csv"),
        player_season = safe_read("player_season.csv"),
        lineups       = safe_read("lineups.csv"),
        formations    = pd.DataFrame(),        # only available from DB
    )
    return _parse_dates(dfs)


@st.cache_data(show_spinner=False)
def load_events(season: str) -> pd.DataFrame:
    """Load events for a single season on demand (~350 MB vs 1 GB for all seasons).

    Results are cached by season so switching views doesn't re-query the DB.
    Uses a WHERE clause so SQLite only reads the relevant index range.
    """
    import os, sqlite3

    if not season:
        return pd.DataFrame()

    # ── Primary path: SQLite ──────────────────────────────────────────────────
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(
                "SELECT * FROM events WHERE season = ?", conn, params=(season,)
            )
            conn.close()
            if "match_date" in df.columns:
                df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()

    # ── Fallback path: CSV ────────────────────────────────────────────────────
    path = f"{PROCESSED}/events.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "season" in df.columns:
        df = df[df["season"] == season].copy()
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


# Show a spinner while data loads, then store everything in D.
with st.spinner("Loading data…"):
    D = load_data()

# If every DataFrame came back empty (neither DB nor CSVs found), stop early.
if all(v.empty for v in D.values()):
    st.error(
        "No data found. Run `python build_processed.py` then `python build_db.py`, "
        "or place CSVs in `data/processed/`."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def team_list(df: pd.DataFrame, *cols: str) -> list[str]:
    """Return a sorted list of unique team names from one or more columns.

    Accepts multiple column names so it works for DataFrames that store the
    team in different fields (e.g. 'team', 'home_team', 'away_team').
    """
    names: set = set()
    for c in cols:
        if c in df.columns:
            names.update(df[c].dropna().unique())
    return sorted(names)


def match_label(row) -> str:
    """Build a human-readable match label from a DataFrame row.

    Expects 'match_date', 'home_team', and 'away_team' columns.
    Used to populate match selector dropdowns across multiple views.
    Example output: '15 Aug  Arsenal v Bournemouth'
    """
    try:
        d = pd.to_datetime(row["match_date"]).strftime("%d %b")
    except Exception:
        d = "?"
    return f"{d}  {row['home_team']} v {row['away_team']}"


def safe_int(val, default=0) -> int:
    """Convert val to int, returning default for NaN/None.

    float('nan') is truthy in Python so `NaN or 0` still returns NaN.
    This helper uses pd.isna() to catch it properly.
    """
    try:
        return default if pd.isna(val) else int(val)
    except (TypeError, ValueError):
        return default


def dark_fig_style(fig, *axes):
    """Apply a consistent dark theme to a matplotlib figure and its axes.

    Called after building any chart so all plots match the dark UI.
    Colours the figure background, axis background, tick labels, and
    axis titles — then re-applies the axis title text if one exists.
    """
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#121212")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        if ax.get_title():
            ax.set_title(ax.get_title(), color="white")


# ---------------------------------------------------------------------------
# Cached grade / predictor helpers
# (defined here — before the view blocks — so they are always reachable
#  regardless of which view is selected)
# ---------------------------------------------------------------------------

@st.cache_data
def _get_team_grades(ms_df: pd.DataFrame):
    """Flatten match_summary and return (season_grades, rolling_grades) DataFrames."""
    if ms_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    flat    = flatten_match_summary(ms_df)
    season  = compute_season_grades(flat)
    rolling = compute_rolling_grades(flat, n=5)
    return season, rolling


@st.cache_data
def _get_player_grades(
    ps_df: pd.DataFrame,
    ev_df: pd.DataFrame,
    lu_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute player grades using Understat, WhoScored events, and ESPN lineups."""
    if ps_df.empty:
        return pd.DataFrame()
    return compute_player_grades(ps_df, events_df=ev_df, lineups_df=lu_df)


@st.cache_data
def _get_poisson_model(crossref_df: pd.DataFrame) -> dict:
    """Build Poisson model from full crossref (all seasons), cached."""
    return build_poisson_model(crossref_df, n_recent=10)


@st.cache_data
def _get_upcoming(crossref_df: pd.DataFrame) -> pd.DataFrame:
    """Return upcoming (unplayed) fixtures from crossref, cached."""
    return get_upcoming_fixtures(crossref_df)


def _team_row(df: pd.DataFrame, team: str) -> pd.Series | None:
    """Return the single row for *team* from a team-indexed DataFrame, or None."""
    rows = df[df["team"] == team]
    return rows.iloc[0] if not rows.empty else None


@st.cache_data
def _flatten_ms(ms: pd.DataFrame) -> pd.DataFrame:
    """Flatten match_summary from wide (one row/match) to long (one row/team/match).

    Returns a DataFrame with columns:
        team, xg_for, xg_against, goals_for, goals_against,
        possession_pct, pass_pct, shot_pct, tackle_pct  (where available)

    Used by xG Performance and Head-to-Head views to avoid duplicating
    the home/away pivot logic.
    """
    if ms.empty:
        return pd.DataFrame()

    home_cols = {"home_team": "team", "home_xg": "xg_for", "away_xg": "xg_against",
                 "home_goals": "goals_for", "away_goals": "goals_against", "home_possession_pct": "possession_pct",
                 "home_pass_pct": "pass_pct", "home_shot_pct": "shot_pct", "home_tackle_pct": "tackle_pct"}
    away_cols = {"away_team": "team", "away_xg": "xg_for", "home_xg": "xg_against",
                 "away_goals": "goals_for", "home_goals": "goals_against", "away_possession_pct": "possession_pct",
                 "away_pass_pct": "pass_pct", "away_shot_pct": "shot_pct", "away_tackle_pct": "tackle_pct"}

    base_home = ms[[c for c in home_cols if c in ms.columns]].rename(columns=home_cols)
    base_away = ms[[c for c in away_cols if c in ms.columns]].rename(columns=away_cols)

    flat = pd.concat([base_home, base_away], ignore_index=True)
    num_cols = [c for c in flat.columns if c != "team"]
    flat[num_cols] = flat[num_cols].apply(pd.to_numeric, errors="coerce")
    return flat


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — SEASON SELECTOR + VIEW SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
# The sidebar is always visible. Selecting a different view re-runs the script
# from top to bottom — Streamlit's execution model means the entire script
# re-executes on every user interaction, but cached data is reused.
#
# Season selector: reads the unique 'season' values from match_summary (the
# most complete dataset).  Falls back to "All Seasons" if the column doesn't
# exist yet (i.e. the user hasn't rebuilt processed data after this update).
# Filtering is applied once here; every view below uses the pre-filtered dict S.

st.sidebar.title("⚽ PL Analytics")
st.sidebar.markdown("---")

# Detect available seasons from the data
_ms = D["match_summary"]
if "season" in _ms.columns and not _ms.empty:
    available_seasons = sorted(_ms["season"].dropna().unique(), reverse=True)
else:
    available_seasons = []

if available_seasons:
    sel_season = st.sidebar.selectbox(
        "Season:",
        available_seasons,
        index=0,   # default to most recent
    )
else:
    sel_season = None
    st.sidebar.info("Season column not found — re-run `build_processed.py` to enable season filtering.")

st.sidebar.markdown("---")

# Apply season filter to every DataFrame.
# S["shots"] etc. are the season-scoped versions used by all views below.
# If sel_season is None (old data format), S is just D unchanged.
def filter_season(df: pd.DataFrame) -> pd.DataFrame:
    if sel_season is None or "season" not in df.columns or df.empty:
        return df
    return df[df["season"] == sel_season].copy()

S = {k: filter_season(v) for k, v in D.items()}

view = st.sidebar.selectbox("View:", [
    "📋 League Table",
    "🎯 Shot Maps",
    "🔥 Event Heatmaps",
    "📊 Match Analysis",
    "👤 Player Stats",
    "📈 xG Analysis",
    "🏅 Team Grades",
    "⭐ Player Grades",
    "🔮 Match Predictor",
    "⚡ Shot Quality",
    "⚔️ Head-to-Head",
    "🗺️ Tactics",
])

st.sidebar.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# 📋  LEAGUE TABLE
# ══════════════════════════════════════════════════════════════════════════════
# Built entirely from match_crossref.csv — no extra source files needed.
# Completed matches: home_goals not NaN.
# Upcoming matches: home_goals is NaN, match_date >= today.
#
# Columns: Pos, Team (with crest), P, W, D, L, GF, GA, GD, Pts, Form, Next

if view == "📋 League Table":
    st.title("📋 League Table")

    mc = D["match_crossref"].copy()

    if mc.empty:
        st.warning("match_crossref data not available — run `build_processed.py`.")
        st.stop()

    # Filter to the season selected in the sidebar
    mc_s = mc[mc["season"] == sel_season].copy() if sel_season else mc.copy()
    st.caption(f"Season: {sel_season or 'All'}")

    completed = mc_s[mc_s["home_goals"].notna()].copy()
    upcoming_all = mc_s[mc_s["home_goals"].isna()].copy()

    if completed.empty:
        st.warning("No completed matches found for this season.")
        st.stop()

    # ── Build per-team records ────────────────────────────────────────────────
    # Flatten wide (one row/match) → long (two rows/match, one per team)
    home_r = completed[["match_date","home_team","away_team","home_goals","away_goals"]].copy()
    home_r = home_r.rename(columns={"home_team":"team","away_team":"opp",
                                     "home_goals":"gf","away_goals":"ga"})
    away_r = completed[["match_date","home_team","away_team","home_goals","away_goals"]].copy()
    away_r = away_r.rename(columns={"away_team":"team","home_team":"opp",
                                     "away_goals":"gf","home_goals":"ga"})
    flat = pd.concat([home_r, away_r], ignore_index=True)
    flat["gf"] = flat["gf"].astype(float)
    flat["ga"] = flat["ga"].astype(float)
    flat["w"]  = (flat["gf"] > flat["ga"]).astype(int)
    flat["d"]  = (flat["gf"] == flat["ga"]).astype(int)
    flat["l"]  = (flat["gf"] < flat["ga"]).astype(int)

    tbl = (
        flat.groupby("team", sort=False)
        .agg(P=("gf","count"), W=("w","sum"), D=("d","sum"), L=("l","sum"),
             GF=("gf","sum"), GA=("ga","sum"))
        .reset_index()
    )
    tbl["GD"]  = (tbl["GF"] - tbl["GA"]).astype(int)
    tbl["Pts"] = (3 * tbl["W"] + tbl["D"]).astype(int)
    tbl["GF"]  = tbl["GF"].astype(int)
    tbl["GA"]  = tbl["GA"].astype(int)
    tbl = tbl.sort_values(["Pts","GD","GF"], ascending=False).reset_index(drop=True)
    tbl.insert(0, "Pos", range(1, len(tbl) + 1))

    # ── Form: last 5 results, most recent rightmost ───────────────────────────
    flat_sorted = flat.sort_values("match_date")
    def _form(team_name: str) -> str:
        recent = flat_sorted[flat_sorted["team"] == team_name].tail(5)
        def _emoji(r):
            if r["w"]: return "🟢"
            if r["d"]: return "⚪"
            return "🔴"
        return "".join(_emoji(r) for _, r in recent.iterrows())

    tbl["Form"] = tbl["team"].apply(_form)

    # ── Next fixture per team ─────────────────────────────────────────────────
    today = pd.Timestamp.now(tz="UTC").normalize()
    upcoming_all["match_date"] = pd.to_datetime(upcoming_all["match_date"], utc=True, errors="coerce")
    future = upcoming_all[upcoming_all["match_date"] >= today].copy()

    def _next(team_name: str) -> str:
        h = future[future["home_team"] == team_name].sort_values("match_date")
        a = future[future["away_team"] == team_name].sort_values("match_date")
        opts = []
        if not h.empty:
            r = h.iloc[0]
            opts.append((r["match_date"], f"{r['match_date'].strftime('%d %b')}  v {r['away_team']} (H)"))
        if not a.empty:
            r = a.iloc[0]
            opts.append((r["match_date"], f"{r['match_date'].strftime('%d %b')}  @ {r['home_team']} (A)"))
        if not opts:
            return "—"
        opts.sort(key=lambda x: x[0])
        return opts[0][1]

    tbl["Next Match"] = tbl["team"].apply(_next)

    # ── Table display with crest in team name ─────────────────────────────────
    # Build a "Team" column with inline HTML crest + name for st.markdown table.
    # The dataframe itself uses plain text; crests are shown above as a header row.
    disp = tbl[["Pos","team","P","W","D","L","GF","GA","GD","Pts","Form","Next Match"]].copy()
    disp = disp.rename(columns={"team": "Team"})

    # Render as HTML table so crests appear inline with team names
    def _team_html(name: str) -> str:
        return logo_html(name, size=20) + name

    _NUM_COLS = {"Pos", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"}
    html_rows = []
    th_cells = []
    for c in disp.columns:
        align = "right" if c in _NUM_COLS else "left"
        th_cells.append(
            f"<th style='padding:6px 10px;text-align:{align};"
            f"color:#aaa;border-bottom:1px solid #333;'>{c}</th>"
        )
    header = "<tr>" + "".join(th_cells) + "</tr>"

    for _, row in disp.iterrows():
        cells = []
        for col in disp.columns:
            val = row[col]
            align = "right" if col in ("Pos","P","W","D","L","GF","GA","GD","Pts") else "left"
            if col == "Team":
                val = _team_html(val)
            # Colour GD green/red
            if col == "GD":
                color = "#00c853" if val > 0 else ("#ff1744" if val < 0 else "#aaa")
                cell = f"<td style='padding:5px 10px;text-align:{align};color:{color};font-weight:600;'>{val:+d}</td>"
            elif col == "Pts":
                cell = f"<td style='padding:5px 10px;text-align:{align};font-weight:700;color:#00ff85;'>{val}</td>"
            elif col == "Pos":
                cell = f"<td style='padding:5px 10px;text-align:{align};color:#888;'>{val}</td>"
            else:
                cell = f"<td style='padding:5px 10px;text-align:{align};'>{val}</td>"
            cells.append(cell)
        html_rows.append("<tr style='border-bottom:1px solid #1e1e1e;'>" + "".join(cells) + "</tr>")

    table_html = f"""
    <div style='overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;background:#0e1117;color:#e8e8e8;font-size:13px;'>
    <thead style='background:#1a1a2e;'>{header}</thead>
    <tbody>{"".join(html_rows)}</tbody>
    </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts: Points bar + GF vs GA scatter ────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Points")
        pts_s = tbl.sort_values("Pts", ascending=True).copy()
        fig_pts = px.bar(
            pts_s, x="Pts", y="team", orientation="h",
            color="Pts", color_continuous_scale=[[0, PURPLE], [1, GREEN]],
            hover_name="team",
            hover_data={"Pts": True, "W": True, "D": True, "L": True,
                        "GF": True, "GA": True, "GD": True, "team": False},
            height=max(350, len(pts_s) * 28),
        )
        fig_pts.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#121212",
            font=dict(color="white"), showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#333", title="Points"),
            yaxis=dict(title="", tickfont=dict(size=11)),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        fig_pts.update_traces(marker_line_width=0)
        st.plotly_chart(fig_pts, use_container_width=True)

    with c2:
        st.subheader("Goals For vs Against")
        tbl_plot = tbl.copy()
        tbl_plot["GD_label"] = tbl_plot["GD"].apply(lambda v: f"{v:+d}")
        max_v = max(tbl_plot["GF"].max(), tbl_plot["GA"].max()) + 5
        fig_gf = px.scatter(
            tbl_plot, x="GF", y="GA",
            color="GD", color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            text="team",
            hover_name="team",
            hover_data={"GF": True, "GA": True, "GD": True,
                        "Pts": True, "W": True, "D": True, "L": True,
                        "team": False},
            height=420,
        )
        # y=x parity line (GF=GA)
        fig_gf.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                         line=dict(color="#555", width=1, dash="dot"))
        fig_gf.update_traces(
            textposition="top center",
            textfont=dict(size=9, color="white"),
            marker=dict(size=11, line=dict(width=0.5, color="#555")),
        )
        fig_gf.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#121212",
            font=dict(color="white"),
            coloraxis_colorbar=dict(title=dict(text="GD", font=dict(color="white")),
                                    tickfont=dict(color="white")),
            xaxis=dict(gridcolor="#333", zerolinecolor="#555", range=[0, max_v]),
            yaxis=dict(gridcolor="#333", zerolinecolor="#555", range=[0, max_v]),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_gf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🎯  SHOT MAPS
# ══════════════════════════════════════════════════════════════════════════════
# Data source: shots.csv (Understat)
# Columns used: team, opponent, player, match_date, x, y, xg, result,
#               shot_type, situation
#
# Coordinate system: opta [0,100]
#   x = 0  →  own goal line (defending end)
#   x = 100 → opponent goal line (attacking end)
#   y = 0  →  top touchline, y = 100 → bottom touchline
# mplsoccer's opta pitch has y=0 at the bottom, so we flip: y_plot = 100 - y

elif view == "🎯 Shot Maps":
    st.title("🎯 Shot Maps")

    shots = S["shots"]
    if shots.empty:
        st.warning("No shot data — run the collection pipeline.")
        st.stop()

    # ── Filters ──────────────────────────────────────────────────────────────
    # Three columns: Team → Player → Match.
    # Each filter narrows the next one — player list only shows players from
    # the selected team; match list only shows matches for that team.
    f1, f2, f3 = st.columns(3)

    with f1:
        sel_team = st.selectbox("Team:", team_list(shots, "team"))
        # Pre-filter the DataFrame to the chosen team so downstream selectors
        # only show relevant options.
        team_shots = shots[shots["team"] == sel_team].copy()

    with f2:
        players = ["All Players"] + sorted(team_shots["player"].dropna().unique())
        sel_player = st.selectbox("Player:", players)

    with f3:
        if "match_date" in team_shots.columns and "opponent" in team_shots.columns:
            # shots.csv stores team + opponent (not home_team / away_team),
            # so we build the label directly from those two columns.
            match_opts = (
                team_shots.dropna(subset=["match_date", "opponent"])
                .assign(label=lambda df:
                    df["match_date"].dt.strftime("%d %b") + "  vs " + df["opponent"])
                .drop_duplicates("label")
                .sort_values("match_date")["label"]
                .tolist()
            )
            sel_match = st.selectbox("Match:", ["All Matches"] + match_opts)
        else:
            sel_match = "All Matches"

    # Apply the chosen filters to produce the final working DataFrame.
    filtered = team_shots.copy()
    if sel_player != "All Players":
        filtered = filtered[filtered["player"] == sel_player]
    if sel_match != "All Matches":
        # Reconstruct the same label string on the filtered rows to compare.
        shot_labels = (
            filtered["match_date"].dt.strftime("%d %b") + "  vs " + filtered["opponent"]
        )
        filtered = filtered[shot_labels == sel_match]

    # Shot result multiselect — lets users hide e.g. blocked shots.
    if "result" in filtered.columns:
        all_results = sorted(filtered["result"].dropna().unique())
        sel_results = st.multiselect("Shot Results:", all_results, default=all_results)
        filtered = filtered[filtered["result"].isin(sel_results)]

    # Toggle between individual shot dots and a KDE density heatmap.
    viz_type = st.radio("Visualization:", ["Shot Map", "Heat Map"], horizontal=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    # Four KPI cards shown above the pitch.
    # Conversion % = goals / total shots × 100.
    m1, m2, m3, m4 = st.columns(4)
    n_shots  = len(filtered)
    n_goals  = int((filtered["result"] == "Goal").sum()) if "result" in filtered.columns else 0
    total_xg = filtered["xg"].sum() if "xg" in filtered.columns else 0.0
    conv_pct = (n_goals / n_shots * 100) if n_shots else 0.0
    m1.metric("Shots",      n_shots)
    m2.metric("Goals",      n_goals)
    m3.metric("xG",         f"{total_xg:.2f}")
    m4.metric("Conversion", f"{conv_pct:.1f}%")

    if filtered.empty:
        st.warning("No shots match the selected filters.")
        st.stop()

    # ── Pitch setup ───────────────────────────────────────────────────────────
    # Pitch() creates a full horizontal pitch.
    # pitch_type="opta" tells mplsoccer the coordinate range is [0,100]×[0,100].
    # pitch.draw() returns a matplotlib figure and axis ready to plot on.
    pitch = Pitch(
        pitch_type="opta",
        pitch_color=PITCH_GREEN, line_color="white",
        line_zorder=2, linewidth=1.5,
    )
    fig, ax = pitch.draw(figsize=(12, 7))

    if viz_type == "Shot Map":
        # Each shot is drawn as a coloured marker sized by xG value.
        # Higher xG = bigger circle, meaning higher-quality chances are more visible.
        # The marker shape also varies by result so colour-blind users can distinguish them.
        RESULT_STYLE = {
            "Goal":        dict(color="lime",    marker="*",  base_s=300),  # star, lime green
            "SavedShot":   dict(color="yellow",  marker="o",  base_s=80),   # circle, yellow
            "BlockedShot": dict(color="orange",  marker="s",  base_s=80),   # square, orange
            "MissedShots": dict(color="#ff4444", marker="x",  base_s=80),   # cross, red
        }
        for _, shot in filtered.iterrows():
            sx  = shot.get("x", 75)
            sy  = 100 - shot.get("y", 50)   # flip y-axis to match mplsoccer opta orientation
            res = shot.get("result", "MissedShots")
            xg  = float(shot.get("xg", 0.05))
            st_ = RESULT_STYLE.get(res, dict(color="grey", marker="x", base_s=60))
            # s= controls marker size: base size + xG scaled up so bigger xG = bigger dot
            pitch.scatter(
                sx, sy,
                s=xg * 600 + st_["base_s"],
                c=st_["color"],
                marker=st_["marker"],
                alpha=0.85,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,        # draw shots on top of the pitch lines
                ax=ax,
            )

        # Build a manual legend using Line2D proxy artists (matplotlib can't
        # auto-generate a legend for scatter calls made through mplsoccer).
        legend_handles = [
            Line2D([0],[0], marker="*", color="w", markerfacecolor="lime",    markersize=13, label="Goal"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="yellow",  markersize=10, label="Saved"),
            Line2D([0],[0], marker="s", color="w", markerfacecolor="orange",  markersize=10, label="Blocked"),
            Line2D([0],[0], marker="x", color="w", markerfacecolor="#ff4444", markersize=10,
                   label="Missed", markeredgecolor="#ff4444"),
        ]
        ax.legend(handles=legend_handles, loc="lower left",
                 facecolor="#1a1a1a", labelcolor="white", framealpha=0.85, fontsize=9)

    else:  # Heat Map
        # KDE (Kernel Density Estimation) draws a smooth density surface showing
        # where shots are most concentrated.  thresh=0.05 hides very sparse outer
        # regions; levels=10 controls the number of contour bands.
        # Needs at least 5 points to produce a meaningful density estimate.
        if len(filtered) >= 5:
            pitch.kdeplot(
                filtered["x"].values,
                (100 - filtered["y"]).values,   # flip y to match opta orientation
                ax=ax, cmap="Reds", fill=True,
                alpha=0.75, thresh=0.05, levels=10,
            )
        # Annotate total shot count in the bottom-centre of the pitch.
        ax.text(50, 5, f"Total Shots: {n_shots}", fontsize=11, ha="center",
               fontweight="bold",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, pad=0.4))

    # Dynamic title — builds up from team → player → match as filters are applied.
    title = sel_team
    if sel_player != "All Players":  title += f" — {sel_player}"
    if sel_match  != "All Matches":  title += f"  ·  {sel_match}"
    ax.set_title(title, fontsize=12, fontweight="bold", color="white", pad=10)
    fig.patch.set_facecolor("#0e1117")
    st.pyplot(fig)
    plt.close(fig)

    # ── Download buttons ──────────────────────────────────────────────────────
    # io.StringIO / io.BytesIO are in-memory file buffers — no temp file on disk.
    dl1, dl2 = st.columns(2)
    with dl1:
        buf = io.StringIO()
        filtered.to_csv(buf, index=False)
        st.download_button("📥 Shot Data (CSV)", buf.getvalue(),
                          file_name="shot_data.csv", mime="text/csv")
    with dl2:
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight", dpi=150)
        st.download_button("📥 Shot Map (PNG)", img.getvalue(),
                          file_name="shot_map.png", mime="image/png")

    # ── Supporting bar charts ─────────────────────────────────────────────────
    # Shot Type: e.g. RightFoot, Header, LeftFoot — how the shot was taken.
    # Situation: e.g. OpenPlay, SetPiece, FromCorner — the game situation.
    # Both use value_counts() which groups and counts each unique string value.
    # barh() draws horizontal bars (easier to read long category labels).
    c1, c2 = st.columns(2)
    with c1:
        if "shot_type" in filtered.columns and filtered["shot_type"].notna().any():
            st.subheader("Shot Type")
            counts = filtered["shot_type"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            ax2.barh(counts.index, counts.values, color=PURPLE)
            ax2.set_xlabel("Shots")
            ax2.grid(axis="x", alpha=0.3)
            dark_fig_style(fig2, ax2)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
    with c2:
        if "situation" in filtered.columns and filtered["situation"].notna().any():
            st.subheader("Situation")
            counts = filtered["situation"].value_counts()
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            ax3.barh(counts.index, counts.values, color=PURPLE)
            ax3.set_xlabel("Shots")
            ax3.grid(axis="x", alpha=0.3)
            dark_fig_style(fig3, ax3)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

    # ── Shot detail table ─────────────────────────────────────────────────────
    # Sorted by xG descending so highest-quality chances appear first.
    # Only columns that actually exist in the data are shown (guards against
    # missing columns if a partial dataset is loaded).
    st.subheader("Shot Details")
    disp = [c for c in ["player", "minute", "result", "xg", "shot_type", "situation"]
            if c in filtered.columns]
    st.dataframe(filtered[disp].sort_values("xg", ascending=False),
                use_container_width=True, height=350, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🔥  EVENT HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
# Data source: events.csv (WhoScored)
# Columns used: team, type, period, x, y, match_date, home_team, away_team
#
# WhoScored records every on-ball action — passes, touches, tackles,
# clearances, interceptions, etc.  This view plots their x/y locations as a
# KDE density heatmap, letting you see where on the pitch a team or player
# concentrates a particular action type.
#
# Period values in this dataset are strings: "FirstHalf", "SecondHalf",
# "PreMatch", "PostGame" — NOT integers 1 and 2.

elif view == "🔥 Event Heatmaps":
    st.title("🔥 Event Heatmaps")

    with st.spinner("Loading events…"):
        events = load_events(sel_season)
    if events.empty:
        st.warning("No event data for this season — run the WhoScored collector.")
        st.stop()

    # Build the list of available event types from whatever is in the data.
    event_types = sorted(events["type"].dropna().unique()) if "type" in events.columns else []

    # ── Row 1: Team → Match ───────────────────────────────────────────────────
    # The match dropdown is built after the team is selected so it only shows
    # fixtures that team actually played.  Column widths [1, 3] make the match
    # selector wider since match labels are longer strings.
    r1a, r1b = st.columns([1, 3])
    with r1a:
        sel_team = st.selectbox("Team:", team_list(events, "team"))

    # Filter to this team and pre-compute match labels once — reused for the
    # match dropdown, the player dropdown, and the match filter below.
    team_events_all = events[events["team"] == sel_team].copy()
    if {"match_date", "home_team", "away_team"}.issubset(team_events_all.columns):
        team_events_all["match_date"] = pd.to_datetime(
            team_events_all["match_date"], errors="coerce"
        )
        team_events_all["_label"] = team_events_all.apply(match_label, axis=1)

    with r1b:
        if "_label" in team_events_all.columns:
            match_opts = (
                team_events_all.dropna(subset=["match_date", "home_team", "away_team"])
                .drop_duplicates("_label")
                .sort_values("match_date", ascending=False)["_label"]
                .tolist()
            )
            sel_match = st.selectbox("Match:", ["All Matches"] + match_opts)
        else:
            sel_match = "All Matches"

    # ── Row 1.5: Player selector ──────────────────────────────────────────────
    # Narrow to the chosen match before building player list so only players
    # who appeared in that game are shown.
    if sel_match != "All Matches" and "_label" in team_events_all.columns:
        ev_for_players = team_events_all[team_events_all["_label"] == sel_match]
    else:
        ev_for_players = team_events_all

    if "player" in ev_for_players.columns:
        player_opts = sorted(
            ev_for_players["player"].dropna().astype(str).unique().tolist()
        )
        sel_player = st.selectbox("Player:", ["All Players"] + player_opts)
    else:
        sel_player = "All Players"

    # ── Row 2: Event types + Period ───────────────────────────────────────────
    r2a, r2b = st.columns([3, 1])
    with r2a:
        default_types = [t for t in ["Pass", "BallTouch", "TackleX"] if t in event_types]
        sel_types = st.multiselect("Event Types:", event_types,
                                   default=default_types or event_types[:3])
    with r2b:
        sel_period = st.selectbox("Period:", ["Full Match", "1st Half", "2nd Half"])

    # ── Apply filters ─────────────────────────────────────────────────────────
    ev = team_events_all.copy()

    if sel_match != "All Matches" and "_label" in ev.columns:
        ev = ev[ev["_label"] == sel_match]

    # Player filter: narrow to a single player when one is chosen.
    if sel_player != "All Players" and "player" in ev.columns:
        ev = ev[ev["player"].astype(str) == sel_player]

    # Event type filter: keep only rows whose 'type' is in the selected list.
    if sel_types:
        ev = ev[ev["type"].isin(sel_types)]

    # Period filter: WhoScored stores "FirstHalf" and "SecondHalf" as strings —
    # map the human-readable dropdown value to the correct string in the data.
    if sel_period != "Full Match" and "period" in ev.columns:
        target = "FirstHalf" if sel_period == "1st Half" else "SecondHalf"
        ev = ev[ev["period"] == target]

    # Drop rows with missing start coordinates — can't plot them.
    valid = ev.dropna(subset=["x", "y"])

    # ── Display mode toggle ───────────────────────────────────────────────────
    disp_mode = st.radio(
        "Display mode:",
        ["🔥 Heatmap", "↗ Arrow Map"],
        horizontal=True,
    )

    # For arrow map, further require end coordinates.
    if disp_mode == "↗ Arrow Map":
        valid_arrows = valid.dropna(subset=["end_x", "end_y"])
        n_plotted = len(valid_arrows)
    else:
        n_plotted = len(valid)

    st.metric("Events plotted", n_plotted)

    if n_plotted == 0:
        st.warning("No events match the selected filters." if len(valid) == 0
                   else "No events with end coordinates — try selecting Pass or Shot types.")
        st.stop()

    # Build a descriptive title from all active filter selections.
    type_str   = ", ".join(sel_types) if sel_types else "All Events"
    match_str  = sel_match if sel_match != "All Matches" else "All Matches"
    player_str = f"  ·  {sel_player}" if sel_player != "All Players" else ""
    title_str  = (
        f"{sel_team}{player_str}  ·  {type_str}  ·  {match_str}  ({sel_period})"
    )

    pitch = Pitch(
        pitch_type="opta",
        pitch_color=PITCH_GREEN, line_color="white",
        line_zorder=2, linewidth=1.5,
    )

    # ── Heatmap branch ────────────────────────────────────────────────────────
    if disp_mode == "🔥 Heatmap":
        fig, ax = pitch.draw(figsize=(12, 7))

        if len(valid) >= 5:
            # kdeplot fits a 2D Kernel Density Estimate over the (x, y)
            # coordinates. cmap="hot": black → red → yellow for dense areas.
            # thresh=0.02 hides the bottom 2% of density (sparse outer edges).
            # WhoScored Opta: x=0→100 (own→opponent goal),
            # y=0→100 (left→right touchline) — matches mplsoccer opta directly.
            pitch.kdeplot(
                valid["x"].values,
                valid["y"].values,
                ax=ax,
                cmap="hot", fill=True,
                alpha=0.72, thresh=0.02, levels=15,
                zorder=2,
            )
        else:
            pitch.scatter(
                valid["x"].values,
                valid["y"].values,
                ax=ax, s=120, color="tomato",
                edgecolors="white", linewidths=0.6,
                zorder=3,
            )
            ax.text(50, 95, f"Only {len(valid)} events — showing scatter",
                   ha="center", fontsize=10, color="white",
                   bbox=dict(boxstyle="round", facecolor="#333", alpha=0.8, pad=0.4))

        ax.set_title(title_str, fontsize=12, fontweight="bold", color="white", pad=12)
        fig.patch.set_facecolor("#0e1117")
        st.pyplot(fig)

    # ── Arrow Map branch ──────────────────────────────────────────────────────
    else:
        fig, ax = pitch.draw(figsize=(12, 7))

        # Compute Euclidean distance in Opta units (0-100 scale).
        dx = valid_arrows["end_x"] - valid_arrows["x"]
        dy = valid_arrows["end_y"] - valid_arrows["y"]
        dist = np.sqrt(dx**2 + dy**2)

        # Fixed arrow width — only the length varies with pass distance.
        ARROW_W = 1.5

        # Color by outcome: green = Successful, red = Unsuccessful / other.
        has_outcome = "outcome_type" in valid_arrows.columns
        if has_outcome:
            colors = np.where(
                valid_arrows["outcome_type"].astype(str).str.lower() == "successful",
                "#44dd88",   # green
                "#ff4b4b",   # red
            )
        else:
            colors = ["#f5a623"] * len(valid_arrows)  # amber fallback

        # Single vectorised call — pitch.arrows accepts full arrays for all
        # positional and colour arguments, so no per-row loop is needed.
        pitch.arrows(
            valid_arrows["x"].values, valid_arrows["y"].values,
            valid_arrows["end_x"].values, valid_arrows["end_y"].values,
            ax=ax,
            color=colors,
            width=ARROW_W,
            headwidth=ARROW_W * 2.5,
            headlength=ARROW_W * 2.0,
            headaxislength=ARROW_W * 1.8,
            alpha=0.75,
            zorder=3,
        )

        # Legend + stats sidebar
        ax.set_title(title_str, fontsize=12, fontweight="bold", color="white", pad=12)
        fig.patch.set_facecolor("#0e1117")

        # Inline legend patches
        legend_handles = []
        if has_outcome:
            legend_handles += [
                mpatches.Patch(color="#44dd88", label="Successful"),
                mpatches.Patch(color="#ff4b4b", label="Unsuccessful"),
            ]
        legend_handles.append(
            mpatches.Patch(color="none",
                           label=f"Avg dist: {dist.mean():.1f}  |  Max: {dist.max():.1f}")
        )
        ax.legend(
            handles=legend_handles,
            loc="lower right", framealpha=0.35,
            fontsize=9, labelcolor="white",
            facecolor="#111", edgecolor="#555",
        )

        st.pyplot(fig)

        # Summary metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg distance", f"{dist.mean():.1f} meters")
        m2.metric("Longest", f"{dist.max():.1f} meters")
        if has_outcome:
            n_succ = (valid_arrows["outcome_type"].astype(str).str.lower() == "successful").sum()
            m3.metric("Success rate", f"{100 * n_succ / len(valid_arrows):.0f}%")

    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight", dpi=150)
    fname = "arrow_map.png" if disp_mode == "↗ Arrow Map" else "event_heatmap.png"
    label = "📥 Arrow Map (PNG)" if disp_mode == "↗ Arrow Map" else "📥 Heatmap (PNG)"
    st.download_button(label, img.getvalue(), file_name=fname, mime="image/png")
    plt.close(fig)

    # ── Event type breakdown bar chart ────────────────────────────────────────
    # Shows a count of every event type for this team/player/period so you can
    # see which actions dominate.  head(20) limits to the top 20 types.
    # The [::-1] reversal puts the most common type at the top of the bar chart.
    st.markdown("---")
    breakdown_subject = sel_player if sel_player != "All Players" else sel_team
    st.subheader(f"{breakdown_subject} — Event Type Breakdown ({sel_period})")
    _bc = team_events_all.copy()
    if sel_match != "All Matches" and "_label" in _bc.columns:
        _bc = _bc[_bc["_label"] == sel_match]
    if sel_player != "All Players" and "player" in _bc.columns:
        _bc = _bc[_bc["player"].astype(str) == sel_player]
    type_counts = (
        _bc
        .pipe(lambda df: df[df["period"] == "FirstHalf"]  if sel_period == "1st Half"
              else (df[df["period"] == "SecondHalf"] if sel_period == "2nd Half" else df))
        ["type"].value_counts().head(20)
    )
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(type_counts.index[::-1], type_counts.values[::-1], color=PURPLE)
    ax2.set_xlabel("Count")
    ax2.grid(axis="x", alpha=0.3)
    dark_fig_style(fig2, ax2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# 📊  MATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
# Data sources: match_summary.csv (Understat xG + forecasts) + shots.csv
# Columns used: home_team, away_team, match_date, home_goals, away_goals,
#               home_xg, away_xg, forecast_win, forecast_draw, forecast_loss

elif view == "📊 Match Analysis":
    st.title("📊 Match Analysis")

    ms = S["match_summary"]
    if ms.empty:
        st.warning("No match data available.")
        st.stop()

    # Sort most recent first so the dropdown defaults to the latest game.
    ms_sorted = ms.sort_values("match_date", ascending=False).copy()
    ms_sorted["_label"] = ms_sorted.apply(match_label, axis=1)

    sel_label = st.selectbox("Select Match:", ms_sorted["_label"].tolist())
    # iloc[0] takes the first (and should be only) row that matches the label.
    row = ms_sorted[ms_sorted["_label"] == sel_label].iloc[0]

    st.markdown("---")

    # ── Match scorecard ───────────────────────────────────────────────────────
    # Three columns: home stats | score | away stats.
    # Column widths [4, 2, 4] give equal space to each team with a narrow
    # centre column just wide enough for the score.
    home_col, mid_col, away_col = st.columns([4, 2, 4])

    with home_col:
        st.markdown(f"### 🏠 {row['home_team']}")
        st.metric("Goals", safe_int(row.get("home_goals")))
        if "home_xg" in row:
            st.metric("xG", f"{row['home_xg']:.2f}")
        if "forecast_win" in row:
            st.metric("Pre-match Win %", f"{row['forecast_win']*100:.0f}%")

    with mid_col:
        # Raw HTML used here because Streamlit's st.metric doesn't support
        # centred large text for a score display.
        date_str = pd.to_datetime(row["match_date"]).strftime("%d %b %Y")
        score = f"{safe_int(row.get('home_goals'))} – {safe_int(row.get('away_goals'))}"
        st.markdown(
            f"<div style='text-align:center; padding-top:24px'>"
            f"<p style='color:#aaa; font-size:13px; margin-bottom:4px'>{date_str}</p>"
            f"<h2 style='color:{GREEN}; margin:0'>{score}</h2>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with away_col:
        st.markdown(f"### ✈️ {row['away_team']}")
        st.metric("Goals", safe_int(row.get("away_goals")))
        if "away_xg" in row:
            st.metric("xG", f"{row['away_xg']:.2f}")
        if "forecast_loss" in row:
            st.metric("Pre-match Win %", f"{row['forecast_loss']*100:.0f}%")

    st.markdown("---")

    # ── xG comparison bar ─────────────────────────────────────────────────────
    # A simple horizontal bar chart showing each team's xG side by side.
    # The numeric label is placed just to the right of each bar end.
    # Home team in green, away in purple — consistent with the colour scheme.
    if {"home_xg", "away_xg"}.issubset(row.index):
        st.subheader("xG Comparison")
        h_xg = row["home_xg"]
        a_xg = row["away_xg"]
        fig, ax = plt.subplots(figsize=(7, 2.2))
        bars = ax.barh(
            [row["away_team"], row["home_team"]],
            [a_xg, h_xg],
            color=[PURPLE, GREEN],
            alpha=0.88,
        )
        # Annotate each bar with its value.
        for bar, val in zip(bars, [a_xg, h_xg]):
            ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
                   f"{val:.2f}", va="center", fontsize=12, fontweight="bold", color="white")
        ax.set_xlabel("Expected Goals (xG)")
        ax.grid(axis="x", alpha=0.3)
        dark_fig_style(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Pre-match forecast ────────────────────────────────────────────────────
    # Understat's forecast probabilities are generated before kick-off from
    # their model.  forecast_win = home win probability, forecast_loss = away
    # win probability (named from the home team's perspective).
    if all(k in row.index for k in ["forecast_win", "forecast_draw", "forecast_loss"]):
        st.subheader("Pre-Match Forecast")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric(f"{row['home_team']} Win", f"{row['forecast_win']*100:.0f}%")
        fc2.metric("Draw",                    f"{row['forecast_draw']*100:.0f}%")
        fc3.metric(f"{row['away_team']} Win", f"{row['forecast_loss']*100:.0f}%")

    # ── Per-match shot map ─────────────────────────────────────────────────────
    # Looks up this match's shots from shots.csv by matching on date and teams.
    # Draws two side-by-side half-pitch views — one per team — using
    # VerticalPitch (the pitch is rotated so the attacking end faces upward).
    # Dot colour: lime=goal, yellow=saved, red=everything else.
    # Dot size scales with xG so higher-quality chances are more visible.
    shots = S["shots"]
    if not shots.empty and "match_date" in shots.columns:
        match_date_val = pd.to_datetime(row["match_date"]).date()
        match_shots = shots[
            (shots["match_date"].dt.date == match_date_val) &
            (shots["team"].isin([row["home_team"], row["away_team"]]))
        ]
        if not match_shots.empty:
            st.markdown("---")
            st.subheader("Shot Map")
            # half=True crops the pitch to the attacking half (x > 50)
            # so all shots appear near the goal and nothing is wasted on
            # empty defensive space.
            pitch = VerticalPitch(
                pitch_type="opta", half=True,
                pitch_color=PITCH_GREEN, line_color="white",
            )
            # nrows=1, ncols=2 draws two pitches side by side in one figure.
            fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(14, 7))
            for ax, team in zip(axes, [row["home_team"], row["away_team"]]):
                t_shots = match_shots[match_shots["team"] == team]
                for _, shot in t_shots.iterrows():
                    res   = shot.get("result", "")
                    color = "lime" if res == "Goal" else ("yellow" if res == "SavedShot" else "#ff4444")
                    pitch.scatter(
                        shot["x"], 100 - shot["y"],   # flip y for correct orientation
                        s=float(shot.get("xg", 0.05)) * 600 + 80,
                        c=color, alpha=0.85,
                        edgecolors="white", linewidths=0.6,
                        zorder=4, ax=ax,
                    )
                xg_total = t_shots["xg"].sum() if "xg" in t_shots.columns else 0
                ax.set_title(
                    f"{team}\n{len(t_shots)} shots  ·  xG {xg_total:.2f}",
                    fontsize=11, fontweight="bold", color="white",
                )
            fig.patch.set_facecolor("#0e1117")
            st.pyplot(fig)
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 👤  PLAYER STATS
# ══════════════════════════════════════════════════════════════════════════════
# Data source: player_season.csv (Understat)
# Columns used: player, primary_team, position, games, goals, assists,
#               shots, key_passes, xg, xa, npg, npxg
#
# 'primary_team' is the player's most recent club — Understat stores
# mid-season transfers as "Bournemouth,Man City" so primary_team takes
# the last entry in that comma-separated string.
# npg = non-penalty goals, npxg = non-penalty expected goals.

elif view == "👤 Player Stats":
    st.title("👤 Player Season Stats")

    ps = S["player_season"]
    if ps.empty:
        st.warning("No player data — run the Understat collector.")
        st.stop()

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        teams = ["All Teams"] + sorted(ps["primary_team"].dropna().unique())
        sel_team = st.selectbox("Team:", teams)
    with f2:
        positions = (["All Positions"] + sorted(ps["position"].dropna().unique())
                     if "position" in ps.columns else ["All Positions"])
        sel_pos = st.selectbox("Position:", positions)
    with f3:
        # Slider filters out players with very few appearances who would skew
        # per-90 stats or appear as outliers in the scatter plots.
        min_games = st.slider("Min. Appearances:", 0, 30, 5)

    filtered = ps.copy()
    if sel_team != "All Teams":
        filtered = filtered[filtered["primary_team"] == sel_team]
    if sel_pos != "All Positions" and "position" in filtered.columns:
        filtered = filtered[filtered["position"] == sel_pos]
    if "games" in filtered.columns:
        filtered = filtered[filtered["games"] >= min_games]

    st.metric("Players shown", len(filtered))

    # ── Stats table ───────────────────────────────────────────────────────────
    # Sorted by xG descending so the most dangerous attackers appear first.
    # Only columns that exist in the DataFrame are included — protects against
    # partial datasets.
    disp_cols = [c for c in
                 ["player", "primary_team", "position", "games", "goals", "assists",
                  "shots", "key_passes", "xg", "xa", "npg", "npxg"]
                 if c in filtered.columns]
    st.dataframe(
        filtered[disp_cols].sort_values("xg", ascending=False),
        use_container_width=True, height=400, hide_index=True,
    )

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Goals vs xG")
        if {"goals", "xg"}.issubset(filtered.columns) and len(filtered) > 1:
            ps_plot = filtered.copy()
            for col in ("xg", "goals"):
                ps_plot[col] = pd.to_numeric(ps_plot[col], errors="coerce")
            ps_plot = ps_plot.dropna(subset=["xg", "goals"])
            ps_plot["diff"] = (ps_plot["goals"] - ps_plot["xg"]).round(2)
            ps_plot["xg"]   = ps_plot["xg"].round(2)
            ps_plot["goals"] = ps_plot["goals"].round(2)
            lim = max(ps_plot["xg"].max(), ps_plot["goals"].max()) * 1.08
            fig_gvx = px.scatter(
                ps_plot, x="xg", y="goals",
                color="xg", color_continuous_scale="RdYlGn",
                hover_name="player",
                hover_data={"primary_team": True, "xg": ":.2f",
                            "goals": ":.0f", "diff": ":.2f",
                            "games": True},
                labels={"xg": "xG", "goals": "Goals",
                        "diff": "Goals − xG", "primary_team": "Team"},
                height=400,
            )
            fig_gvx.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                              line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dot"))
            fig_gvx.update_traces(marker=dict(size=7, line=dict(width=0.4, color="white")))
            fig_gvx.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#121212",
                font=dict(color="white"), coloraxis_showscale=False,
                xaxis=dict(gridcolor="#333", range=[0, lim]),
                yaxis=dict(gridcolor="#333", range=[0, lim]),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_gvx, use_container_width=True)

    with c2:
        st.subheader("Top 15 by xG")
        if "xg" in filtered.columns:
            top15 = filtered.nlargest(15, "xg").copy()
            top15["xg"] = pd.to_numeric(top15["xg"], errors="coerce").round(2)
            hover_cols = {c: True for c in ["primary_team", "goals", "games"] if c in top15.columns}
            hover_cols["xg"] = ":.2f"
            fig_top = px.bar(
                top15.sort_values("xg"), x="xg", y="player", orientation="h",
                color="xg", color_continuous_scale=[[0, PURPLE], [1, GREEN]],
                hover_name="player",
                hover_data=hover_cols,
                labels={"xg": "xG", "player": ""},
                height=420,
            )
            fig_top.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#121212",
                font=dict(color="white"), coloraxis_showscale=False,
                xaxis=dict(gridcolor="#333", title="xG"),
                yaxis=dict(title="", tickfont=dict(size=10)),
                margin=dict(t=10, b=10),
            )
            fig_top.update_traces(marker_line_width=0)
            st.plotly_chart(fig_top, use_container_width=True)

    if {"assists", "xa"}.issubset(filtered.columns) and len(filtered) > 1:
        st.markdown("---")
        st.subheader("Assists vs xA")
        xa_plot = filtered.copy()
        for col in ("xa", "assists"):
            xa_plot[col] = pd.to_numeric(xa_plot[col], errors="coerce")
        xa_plot = xa_plot.dropna(subset=["xa", "assists"])
        xa_plot["diff"] = (xa_plot["assists"] - xa_plot["xa"]).round(2)
        xa_plot["xa"]      = xa_plot["xa"].round(2)
        xa_plot["assists"] = xa_plot["assists"].round(2)
        lim_xa = max(xa_plot["xa"].max(), xa_plot["assists"].max()) * 1.08
        fig_xavs = px.scatter(
            xa_plot, x="xa", y="assists",
            color="xa", color_continuous_scale="Blues",
            hover_name="player",
            hover_data={"primary_team": True, "xa": ":.2f",
                        "assists": ":.0f", "diff": ":.2f", "games": True},
            labels={"xa": "xA", "assists": "Assists",
                    "diff": "Assists − xA", "primary_team": "Team"},
            height=380,
        )
        fig_xavs.add_shape(type="line", x0=0, y0=0, x1=lim_xa, y1=lim_xa,
                           line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dot"))
        fig_xavs.update_traces(marker=dict(size=7, line=dict(width=0.4, color="white")))
        fig_xavs.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#121212",
            font=dict(color="white"), coloraxis_showscale=False,
            xaxis=dict(gridcolor="#333", range=[0, lim_xa]),
            yaxis=dict(gridcolor="#333", range=[0, lim_xa]),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_xavs, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 📈  xG ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
# Data source: match_summary.csv (Understat)
# Aggregates match-level xG into team-season totals for league-wide comparison.
#
# xGF = Expected Goals For  (quality of chances created)
# xGA = Expected Goals Against (quality of chances conceded)
# xGD = xGF − xGA  (net expected goal difference — proxy for true team quality)
# G − xG = actual goals minus xG  (positive = overperforming the model)

elif view == "📈 xG Analysis":
    st.title("📈 xG Analysis")

    ms = S["match_summary"]
    ps = S["player_season"].copy()
    needed = {"home_team", "away_team", "home_xg", "away_xg", "home_goals", "away_goals"}
    if ms.empty or not needed.issubset(ms.columns):
        st.warning("Match summary data is incomplete — re-run `build_processed.py`.")
        st.stop()

    # ── Build team-level season aggregates ────────────────────────────────────
    flat = _flatten_ms(ms).dropna(subset=["xg_for", "xg_against", "goals_for", "goals_against"])
    agg = flat.groupby("team")[["xg_for", "xg_against", "goals_for", "goals_against"]].sum().reset_index()
    agg = agg[agg["xg_for"] > 0].copy()
    agg["xgd"]          = agg["xg_for"]    - agg["xg_against"]
    agg["attack_diff"]  = agg["goals_for"] - agg["xg_for"]
    agg["defense_diff"] = agg["xg_against"] - agg["goals_against"]
    agg = agg.sort_values("xg_for", ascending=False).reset_index(drop=True)

    tab_team, tab_player = st.tabs(["🏟️ Team xG", "👤 Player xG"])

    # ══════════════════════════════════════════════════════════════════════
    with tab_team:

        # ── xGF vs xGA interactive scatter ────────────────────────────────
        # Quadrant chart: top-right = dominant, bottom-left = defensive,
        # top-left = struggling, bottom-right = open/exciting.
        # Colour encodes xGD (green = positive, red = negative).
        # Hover tooltip shows full stats per team.
        st.subheader("xG For vs xG Against")
        st.caption("Hover over a point to see team stats  ·  Dashed lines = league average  ·  Y-axis inverted: low xGA (good defence) floats to the top")

        avg_xgf = agg["xg_for"].mean()
        avg_xga = agg["xg_against"].mean()

        fig_scatter = px.scatter(
            agg.round(2),
            x="xg_for", y="xg_against",
            color="xgd",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            text="team",
            hover_name="team",
            hover_data={
                "xg_for":       ":.2f",
                "xg_against":   ":.2f",
                "goals_for":    ":.0f",
                "goals_against":":.0f",
                "xgd":          ":.2f",
                "attack_diff":  ":.2f",
                "defense_diff": ":.2f",
                "team":         False,
            },
            labels={
                "xg_for":       "xG For",
                "xg_against":   "xG Against",
                "xgd":          "xGD",
                "goals_for":    "Goals",
                "goals_against":"Goals Against",
                "attack_diff":  "Atk Over/Under",
                "defense_diff": "Def Over/Under",
            },
            height=550,
        )
        fig_scatter.update_traces(
            textposition="top center",
            textfont=dict(size=10, color="white"),
            marker=dict(size=12, line=dict(width=1, color="white")),
        )
        fig_scatter.add_hline(y=avg_xga, line_dash="dot", line_color="rgba(255,255,255,0.35)", line_width=1)
        fig_scatter.add_vline(x=avg_xgf, line_dash="dot", line_color="rgba(255,255,255,0.35)", line_width=1)
        fig_scatter.update_yaxes(autorange="reversed")  # low xGA (good defence) at top
        fig_scatter.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#121212",
            font=dict(color="white"),
            coloraxis_colorbar=dict(title=dict(text="xGD", font=dict(color="white")), tickfont=dict(color="white")),
            xaxis=dict(gridcolor="#333", zerolinecolor="#555"),
            yaxis=dict(gridcolor="#333", zerolinecolor="#555"),
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")

        # ── Over/underperformance bars ─────────────────────────────────────
        col_atk, col_def = st.columns(2)

        with col_atk:
            st.subheader("Attacking Over/Underperformance")
            st.caption("Goals − xG  ·  Green = scoring above model, Red = below")
            atk = agg.sort_values("attack_diff", ascending=True).copy()
            atk["perf"] = atk["attack_diff"].apply(lambda v: "Over" if v >= 0 else "Under")
            fig_atk = px.bar(
                atk, x="attack_diff", y="team",
                orientation="h",
                color="perf",
                color_discrete_map={"Over": GREEN, "Under": "#e74c3c"},
                hover_name="team",
                hover_data={"attack_diff": ":.2f", "xg_for": ":.2f", "goals_for": ":.0f", "perf": False},
                labels={"attack_diff": "Goals − xG", "team": "", "perf": "",
                        "xg_for": "xGF", "goals_for": "GF"},
                height=max(350, len(atk) * 25),
            )
            fig_atk.add_vline(x=0, line_color="white", line_dash="dash", line_width=1)
            fig_atk.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"), showlegend=False,
                xaxis=dict(gridcolor="#333", zerolinecolor="#555"),
                yaxis=dict(gridcolor="#333"),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_atk, use_container_width=True)

        with col_def:
            st.subheader("Defensive Over/Underperformance")
            st.caption("xGA − Goals Conceded  ·  Green = conceding less than model, Red = leaky")
            dfn = agg.sort_values("defense_diff", ascending=True).copy()
            dfn["perf"] = dfn["defense_diff"].apply(lambda v: "Over" if v >= 0 else "Under")
            fig_def = px.bar(
                dfn, x="defense_diff", y="team",
                orientation="h",
                color="perf",
                color_discrete_map={"Over": GREEN, "Under": "#e74c3c"},
                hover_name="team",
                hover_data={"defense_diff": ":.2f", "xg_against": ":.2f", "goals_against": ":.0f", "perf": False},
                labels={"defense_diff": "xGA − Goals Conceded", "team": "", "perf": "",
                        "xg_against": "xGA", "goals_against": "GA"},
                height=max(350, len(dfn) * 25),
            )
            fig_def.add_vline(x=0, line_color="white", line_dash="dash", line_width=1)
            fig_def.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"), showlegend=False,
                xaxis=dict(gridcolor="#333", zerolinecolor="#555"),
                yaxis=dict(gridcolor="#333"),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_def, use_container_width=True)

        st.markdown("---")

        # ── Expected vs Actual Goals grouped bar ──────────────────────────
        st.subheader("Expected vs Actual Goals — all teams")
        agg_sorted = agg.sort_values("xg_for", ascending=False)
        agg_melt = agg_sorted.melt(
            id_vars=["team", "xg_against", "goals_against", "xgd", "attack_diff", "defense_diff"],
            value_vars=["xg_for", "goals_for"],
            var_name="metric", value_name="value",
        )
        agg_melt["metric"] = agg_melt["metric"].map({"xg_for": "xGF", "goals_for": "Goals"})
        fig_bar = px.bar(
            agg_melt, x="team", y="value", color="metric",
            barmode="group",
            color_discrete_map={"xGF": PURPLE, "Goals": GREEN},
            hover_data={"value": ":.2f", "metric": False},
            labels={"value": "Goals / xG", "team": "", "metric": ""},
            height=420,
        )
        fig_bar.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(tickangle=-45, gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            legend=dict(bgcolor="#1a1a1a", font=dict(color="white")),
            margin=dict(t=10, b=80),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # ── Summary table ─────────────────────────────────────────────────
        st.subheader("Full Team Summary")
        st.dataframe(
            agg.rename(columns={
                "xg_for":       "xGF",
                "goals_for":    "GF",
                "xg_against":   "xGA",
                "goals_against":"GA",
                "xgd":          "xGD",
                "attack_diff":  "Atk Over/Under",
                "defense_diff": "Def Over/Under",
            }).sort_values("xGD", ascending=False).round(2),
            use_container_width=True, hide_index=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    with tab_player:
        if ps.empty:
            st.warning("No player season data for this season.")
            st.stop()

        needed_ps = {"player", "team", "games", "npxg", "npg"}
        if not needed_ps.issubset(ps.columns):
            st.warning(f"player_season missing columns: {needed_ps - set(ps.columns)}")
            st.stop()

        ps_f = ps.copy()
        for col in ("games", "npxg", "npg"):
            ps_f[col] = pd.to_numeric(ps_f[col], errors="coerce").fillna(0)
        ps_f = ps_f[ps_f["games"] >= 3]
        if "position" in ps_f.columns:
            ps_f = ps_f[~ps_f["position"].astype(str).str.contains("GK", na=False)]
        if ps_f.empty:
            st.warning("No qualifying players (≥ 3 games, non-GK) found.")
            st.stop()

        ps_f["diff"] = (ps_f["npg"] - ps_f["npxg"]).round(2)
        ps_f["npxg"] = ps_f["npxg"].round(2)
        ps_f["npg"]  = ps_f["npg"].round(2)

        # ── Interactive scatter with hover tooltips ────────────────────────
        st.subheader("Non-Penalty Goals vs npxG")
        st.caption("Hover over a point to see player details  ·  Above the line = scoring more than expected")

        lim_max = max(ps_f["npxg"].max(), ps_f["npg"].max()) * 1.08
        lim_max = max(float(lim_max), 1.0)

        fig_ps = px.scatter(
            ps_f,
            x="npxg", y="npg",
            color="team",
            hover_name="player",
            hover_data={
                "team":  True,
                "npxg":  ":.2f",
                "npg":   ":.2f",
                "diff":  ":.2f",
                "games": ":.0f",
            },
            labels={
                "npxg": "Non-Penalty xG",
                "npg":  "Non-Penalty Goals",
                "diff": "Goals − npxG",
                "games":"Games",
            },
            height=580,
            opacity=0.82,
        )
        # y = x reference line
        fig_ps.add_shape(
            type="line", x0=0, y0=0, x1=lim_max, y1=lim_max,
            line=dict(color="rgba(255,255,255,0.5)", width=1.5, dash="dot"),
        )
        fig_ps.add_annotation(
            x=lim_max * 0.72, y=lim_max * 0.93,
            text="Clinical ▲", showarrow=False,
            font=dict(color="rgba(255,255,255,0.55)", size=10),
        )
        fig_ps.add_annotation(
            x=lim_max * 0.68, y=lim_max * 0.07,
            text="Wasteful ▼", showarrow=False,
            font=dict(color="rgba(255,255,255,0.55)", size=10),
        )
        fig_ps.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        fig_ps.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#121212",
            font=dict(color="white"),
            xaxis=dict(range=[0, lim_max], gridcolor="#333", zerolinecolor="#555"),
            yaxis=dict(range=[0, lim_max], gridcolor="#333", zerolinecolor="#555"),
            legend=dict(bgcolor="rgba(30,30,30,0.8)", bordercolor="#555",
                        font=dict(size=9), title_font=dict(size=10)),
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig_ps, use_container_width=True)

        # ── Top 5 over / underperformers ──────────────────────────────────
        col_ov, col_un = st.columns(2)
        over  = ps_f.nlargest(5,  "diff")[["player", "team", "npxg", "npg", "diff"]]
        under = ps_f.nsmallest(5, "diff")[["player", "team", "npxg", "npg", "diff"]]
        with col_ov:
            st.subheader("Top 5 Overperformers")
            st.dataframe(over.rename(columns={"diff": "Goals − npxG"}),
                         use_container_width=True, hide_index=True)
        with col_un:
            st.subheader("Top 5 Underperformers")
            st.dataframe(under.rename(columns={"diff": "Goals − npxG"}),
                         use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🏅  TEAM GRADES
# ══════════════════════════════════════════════════════════════════════════════
# Grades are computed from match_summary.csv (ESPN + Understat metrics).
# Each sub-grade is scaled 1-10 relative to the season's competition:
#   Attack  — xG, goals, shots on target %, xG per shot
#   Defense — xGA (inverted), goals conceded (inverted), saves, interceptions
#   Style   — possession %, pass accuracy, tackle success %
#   Overall — Attack 40% + Defense 40% + Style 20%
#
# "Season Grade" = season-long average.
# "Rolling Grade" = form going into each match (trailing 5-match window).

elif view == "🏅 Team Grades":
    st.title("🏅 Team Grades")

    ms = D["match_summary"]   # Use full dataset (all seasons); filter below
    if ms.empty:
        st.warning("No match summary data — run `build_processed.py`.")
        st.stop()

    with st.spinner("Computing grades…"):
        season_grades, rolling_grades = _get_team_grades(ms)

    if season_grades.empty:
        st.warning("Could not compute grades — check match_summary.csv.")
        st.stop()

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2 = st.columns(2)
    with f1:
        # Season filter scoped to grades data (may span multiple seasons)
        avail_seasons = sorted(season_grades["season"].dropna().unique(), reverse=True)
        sel_gs = st.selectbox("Season:", avail_seasons, key="gs_season")
    with f2:
        # Team selector filtered to the chosen season
        season_teams = sorted(
            season_grades[season_grades["season"] == sel_gs]["team"].dropna().unique()
        )
        sel_gt = st.selectbox("Team:", season_teams, key="gs_team")

    # ── Season grade card for selected team ───────────────────────────────────
    st.markdown("---")
    st.markdown(
        logo_html(sel_gt, size=28) + f"&nbsp;<b style='font-size:18px'>{sel_gt} — Season Grades ({sel_gs})</b>",
        unsafe_allow_html=True,
    )

    team_row = season_grades[
        (season_grades["season"] == sel_gs) & (season_grades["team"] == sel_gt)
    ]

    if not team_row.empty:
        r = team_row.iloc[0]

        # Four metric cards: attack / defense / style / overall
        # Color helper: red ≤ 4, amber 4-6, green ≥ 7
        def _grade_color(g):
            if g >= 7:   return "#00c853"   # green
            if g >= 4.5: return "#ffd600"   # amber
            return "#ff1744"                # red

        c1, c2, c3, c4 = st.columns(4)
        for col, label, key in [
            (c1, "⚔️ Attack",   "attack_grade"),
            (c2, "🛡️ Defense",  "defense_grade"),
            (c3, "🎨 Style",    "style_grade"),
            (c4, "⭐ Overall",  "overall_grade"),
        ]:
            val = round(float(r[key]), 1)
            color = _grade_color(val)
            col.markdown(
                f"<div style='background:#1e1e1e;border-left:4px solid {color};"
                f"padding:14px 18px;border-radius:6px;'>"
                f"<div style='font-size:13px;color:#aaa;'>{label}</div>"
                f"<div style='font-size:32px;font-weight:700;color:{color};'>{val:.1f}</div>"
                f"<div style='font-size:11px;color:#666;'>/ 10.0</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Record summary ────────────────────────────────────────────────────
        st.markdown("")
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric("Matches",  int(r["matches_played"]))
        rc2.metric("Wins",     int(r["wins"]))
        rc3.metric("Draws",    int(r["draws"]))
        rc4.metric("Losses",   int(r["losses"]))
        rc5.metric("Avg Pts",  f"{r['avg_points']:.2f}")

    # ── Rolling form chart (Overall grade, last 10+ GWs) ──────────────────────
    st.markdown("---")
    st.subheader(f"Rolling Form — {sel_gt} ({sel_gs})")

    # Filter rolling grades to selected team + season
    team_rolling = rolling_grades[
        (rolling_grades["season"] == sel_gs) & (rolling_grades["team"] == sel_gt)
    ].copy()

    if not team_rolling.empty and "roll_overall_grade" in team_rolling.columns:
        # Sort by match date for a left-to-right timeline
        team_rolling = team_rolling.sort_values("match_date").reset_index(drop=True)
        team_rolling["gw"] = range(1, len(team_rolling) + 1)   # gameweek number

        fig_form = go.Figure()

        grade_lines = [
            ("roll_overall_grade", "⭐ Overall",  "#ffffff", 2.5),
            ("roll_attack_grade",  "⚔️ Attack",   "#00ff85", 1.5),
            ("roll_defense_grade", "🛡️ Defense",  "#4da6ff", 1.5),
            ("roll_style_grade",   "🎨 Style",    "#ffd700", 1.2),
        ]
        has_result = "result" in team_rolling.columns
        for col, lbl, color, lw in grade_lines:
            if col in team_rolling.columns:
                cdata = (
                    list(zip(team_rolling["result"], team_rolling[col]))
                    if has_result
                    else list(zip([""] * len(team_rolling), team_rolling[col]))
                )
                fig_form.add_trace(go.Scatter(
                    x=team_rolling["gw"],
                    y=team_rolling[col],
                    mode="lines+markers",
                    name=lbl,
                    line=dict(color=color, width=lw),
                    marker=dict(size=5),
                    customdata=cdata,
                    hovertemplate=(
                        f"<b>{lbl}</b><br>"
                        "GW: %{x}<br>"
                        "Grade: %{customdata[1]:.2f}<br>"
                        "Result: %{customdata[0]}<extra></extra>"
                    ),
                ))

        fig_form.add_hrect(y0=7, y1=10, fillcolor="#00c853", opacity=0.06, line_width=0)
        fig_form.add_hrect(y0=1, y1=4,  fillcolor="#ff1744", opacity=0.06, line_width=0)
        fig_form.add_hline(y=5.5, line_color="white", line_dash="dash",
                           line_width=0.5, opacity=0.3)

        if has_result:
            result_colors = {"W": "#00c853", "D": "#ffd700", "L": "#ff1744"}
            for _, rrow in team_rolling.iterrows():
                res = rrow.get("result", "")
                if res in result_colors:
                    fig_form.add_annotation(
                        x=rrow["gw"], y=1.2, text=f"<b>{res}</b>",
                        showarrow=False,
                        font=dict(color=result_colors[res], size=9),
                        yref="y",
                    )

        fig_form.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(title="Gameweek", gridcolor="#333",
                       range=[1, len(team_rolling)]),
            yaxis=dict(title="Grade (1–10)", gridcolor="#333", range=[1, 10]),
            legend=dict(bgcolor="#1a1a1a", font=dict(color="white")),
            height=380,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_form, use_container_width=True)
    else:
        st.info("Not enough rolling data for this team / season.")

    # ── All-teams season grade table ──────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"All Teams — {sel_gs}")

    season_tbl = season_grades[season_grades["season"] == sel_gs].copy()
    season_tbl = season_tbl.sort_values("overall_grade", ascending=False).reset_index(drop=True)

    # Display columns
    disp = ["team", "matches_played", "wins", "draws", "losses",
            "avg_points", "avg_xg", "avg_xga",
            "attack_grade", "defense_grade", "style_grade", "overall_grade"]
    disp = [c for c in disp if c in season_tbl.columns]

    # Round float columns for cleaner display
    for c in ["avg_points", "avg_xg", "avg_xga",
              "attack_grade", "defense_grade", "style_grade", "overall_grade"]:
        if c in season_tbl.columns:
            season_tbl[c] = season_tbl[c].round(1)

    st.dataframe(
        season_tbl[disp].rename(columns={
            "team": "Team", "matches_played": "MP", "wins": "W",
            "draws": "D", "losses": "L", "avg_points": "Avg Pts",
            "avg_xg": "Avg xGF", "avg_xga": "Avg xGA",
            "attack_grade": "Attack", "defense_grade": "Defense",
            "style_grade": "Style", "overall_grade": "Overall",
        }),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ⭐  PLAYER GRADES
# ══════════════════════════════════════════════════════════════════════════════
# Grades are derived from player_season.csv (Understat per-season xG/xA data).
# Scaled 1-10 within season so grades reflect quality relative to same-season peers.
#
#   Attack grade     — npxG per 90 min (non-penalty xG)
#   Creativity grade — xA per 90 (60%) + key passes per 90 (40%)
#   Overall grade    — Attack 50% + Creativity 30% + xG Chain 20%
#
# Only players with ≥ 90 minutes played are included.

elif view == "⭐ Player Grades":
    st.title("⭐ Player Grades")

    ps = D["player_season"]
    if ps.empty:
        st.warning("No player data — run the Understat collector.")
        st.stop()

    ps_for_grades = (
        ps[ps["season"] == sel_season].copy()
        if "season" in ps.columns and sel_season
        else ps
    )
    with st.spinner("Computing player grades…"):
        pg = _get_player_grades(ps_for_grades, load_events(sel_season), D["lineups"])

    if pg.empty:
        st.warning("Could not compute player grades.")
        st.stop()

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        avail_seasons = sorted(pg["season"].dropna().unique(), reverse=True)
        sel_pgs = st.selectbox("Season:", avail_seasons, key="pg_season")
    with f2:
        teams_in_season = ["All Teams"] + sorted(
            pg[pg["season"] == sel_pgs]["team"].dropna().unique()
        )
        sel_pgt = st.selectbox("Team:", teams_in_season, key="pg_team")
    with f3:
        positions = ["All Positions"] + sorted(
            pg[pg["season"] == sel_pgs]["position"].dropna().unique()
        ) if "position" in pg.columns else ["All Positions"]
        sel_pgp = st.selectbox("Position:", positions, key="pg_pos")
    with f4:
        min_mins = st.slider("Min. Minutes:", 90, 2000, 450, step=90, key="pg_mins")

    # Apply filters
    pgf = pg[pg["season"] == sel_pgs].copy()
    if sel_pgt != "All Teams":
        pgf = pgf[pgf["team"] == sel_pgt]
    if sel_pgp != "All Positions" and "position" in pgf.columns:
        pgf = pgf[pgf["position"] == sel_pgp]
    pgf = pgf[pgf["time"] >= min_mins]

    st.metric("Players shown", len(pgf))

    # ── Grade table ───────────────────────────────────────────────────────────
    # Sorted by overall_grade descending; color map applied to grade columns.
    # Show defensive columns for DEF/MID, saves for GK, offensive for FWD/MID
    disp_cols = [c for c in
                 ["player", "team", "pos_group", "games", "time",
                  "goals", "assists",
                  "npxg_p90", "xa_p90", "kp_p90",
                  "tackles_won_p90", "interceptions_p90", "clearances_p90",
                  "saves_p90",
                  "attack_grade", "creativity_grade", "defensive_grade", "overall_grade"]
                 if c in pgf.columns]

    tbl = pgf[disp_cols].sort_values("overall_grade", ascending=False).copy()

    # Round all numeric grade/metric columns for clean display
    for c in ["npxg_p90", "xa_p90", "kp_p90",
              "tackles_won_p90", "interceptions_p90", "clearances_p90", "saves_p90",
              "attack_grade", "creativity_grade", "defensive_grade", "overall_grade"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].round(2)

    st.dataframe(
        tbl.rename(columns={
            "player": "Player", "team": "Team", "pos_group": "Pos",
            "games": "Apps", "time": "Mins",
            "goals": "Goals", "assists": "Assists",
            "npxg_p90": "npxG/90", "xa_p90": "xA/90", "kp_p90": "KP/90",
            "tackles_won_p90": "Tkl Won/90", "interceptions_p90": "Int/90",
            "clearances_p90": "Clr/90", "saves_p90": "Saves/90",
            "attack_grade": "Attack", "creativity_grade": "Creativity",
            "defensive_grade": "Defense", "overall_grade": "Overall",
        }),
        use_container_width=True, hide_index=True, height=450,
    )

    st.markdown("---")

    # ── Top 15 by overall grade (bar chart) ───────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Top 15 by Overall Grade")
        top15 = pgf.nlargest(15, "overall_grade").copy()
        if not top15.empty:
            top15["grade_band"] = top15["overall_grade"].apply(
                lambda g: "Elite (7+)" if g >= 7 else ("Average (4.5–7)" if g >= 4.5 else "Below (< 4.5)")
            )
            hover_t15 = {"overall_grade": ":.2f", "grade_band": False}
            for col in ["team", "pos_group", "attack_grade", "creativity_grade", "defensive_grade"]:
                if col in top15.columns:
                    hover_t15[col] = True
            fig_pg_bar = px.bar(
                top15.sort_values("overall_grade", ascending=True),
                x="overall_grade", y="player",
                orientation="h",
                color="grade_band",
                color_discrete_map={
                    "Elite (7+)": "#00c853",
                    "Average (4.5–7)": "#ffd600",
                    "Below (< 4.5)": "#ff1744",
                },
                hover_name="player",
                hover_data=hover_t15,
                labels={"overall_grade": "Overall Grade (1–10)", "player": "",
                        "pos_group": "Pos", "team": "Team",
                        "attack_grade": "Attack", "creativity_grade": "Creativity",
                        "defensive_grade": "Defense"},
                height=460,
            )
            fig_pg_bar.add_vline(x=5.5, line_color="white", line_dash="dash",
                                 line_width=1, opacity=0.3)
            fig_pg_bar.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"), showlegend=False,
                xaxis=dict(range=[1, 10], gridcolor="#333"),
                yaxis=dict(gridcolor="#333"),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_pg_bar, use_container_width=True)

    with c2:
        # ── Attack vs Defense scatter ─────────────────────────────────────────
        # Outfield players only (GKs have NaN attack_grade).
        # Four quadrants: top-right = complete players, top-left = defensive
        # specialists, bottom-right = pure attackers, bottom-left = fringe.
        # Dashed lines mark the group average for each axis.
        st.subheader("Attack vs Defense")
        outfield = pgf[pgf["pos_group"] != "GK"].dropna(
            subset=["attack_grade", "defensive_grade"]
        )
        if len(outfield) > 2:
            hover_of = {
                "team": True, "pos_group": True,
                "attack_grade": ":.2f", "defensive_grade": ":.2f",
                "overall_grade": ":.2f",
            }
            if "creativity_grade" in outfield.columns:
                hover_of["creativity_grade"] = ":.2f"
            fig_pg_sc = px.scatter(
                outfield,
                x="attack_grade", y="defensive_grade",
                color="overall_grade",
                color_continuous_scale="RdYlGn",
                hover_name="player",
                hover_data=hover_of,
                labels={
                    "attack_grade": "Attack Grade",
                    "defensive_grade": "Defense Grade",
                    "pos_group": "Position", "team": "Team",
                    "overall_grade": "Overall",
                    "creativity_grade": "Creativity",
                },
                range_x=[1, 10], range_y=[1, 10],
                height=460,
                opacity=0.82,
            )
            fig_pg_sc.add_hline(
                y=outfield["defensive_grade"].mean(),
                line_color="white", line_dash="dash", line_width=1, opacity=0.3,
            )
            fig_pg_sc.add_vline(
                x=outfield["attack_grade"].mean(),
                line_color="white", line_dash="dash", line_width=1, opacity=0.3,
            )
            fig_pg_sc.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"),
                coloraxis_colorbar=dict(
                    title=dict(text="Overall", font=dict(color="white")),
                    tickfont=dict(color="white"),
                ),
                xaxis=dict(gridcolor="#333"),
                yaxis=dict(gridcolor="#333"),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_pg_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🔮  MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
# Uses a Poisson goals model (Dixon-Coles style) to predict W/D/L probabilities
# for any upcoming fixture.
#
# Model inputs (from match_crossref.csv):
#   - Each team's attack rating = avg xG scored / league avg, last 10 matches
#   - Each team's defense rating = avg xGA / league avg, last 10 matches
#   - Home advantage = ratio of total home xG to away xG
#   - Ratings shrunk toward 1.0 (league avg) via Bayesian prior
#
# Lambda (expected goals):
#   home = league_avg × attack(home) × defense(away) × home_advantage
#   away = league_avg × attack(away) × defense(home)
#
# W/D/L probabilities are summed from a 7×7 Poisson scoreline grid.

elif view == "🔮 Match Predictor":
    st.title("🔮 Match Predictor")
    st.caption(
        "Poisson model using each team's last 10 matches (xG-based attack/defense ratings). "
        "Grades reflect form going into the fixture."
    )

    crossref = D["match_crossref"]
    if crossref.empty:
        st.warning("No crossref data — run `build_processed.py`.")
        st.stop()

    with st.spinner("Building model…"):
        poisson_model = _get_poisson_model(crossref)
        upcoming      = _get_upcoming(crossref)
        season_grades, _ = _get_team_grades(D["match_summary"])

    if upcoming.empty:
        st.info("No upcoming fixtures found in the data. "
                "Re-run the ESPN collector to refresh the schedule.")
        st.stop()

    # ── Fixture selector ─────────────────────────────────────────────────────
    # Show all upcoming fixtures in a dropdown, grouped implicitly by date.
    fixture_labels = upcoming["fixture_label"].tolist()
    sel_fixture_lbl = st.selectbox("Select Fixture:", fixture_labels)

    fixture_row = upcoming[upcoming["fixture_label"] == sel_fixture_lbl].iloc[0]
    home_team = fixture_row["home_team"]
    away_team = fixture_row["away_team"]
    match_date_str = pd.to_datetime(fixture_row["match_date"]).strftime("%A %-d %B %Y")

    st.markdown(
        logo_html(home_team, size=32) + f"&nbsp;<b>{home_team}</b>"
        + "&nbsp;&nbsp;vs&nbsp;&nbsp;"
        + logo_html(away_team, size=32) + f"&nbsp;<b>{away_team}</b>",
        unsafe_allow_html=True,
    )
    st.caption(f"📅 {match_date_str}")
    st.markdown("---")

    # ── Run prediction ────────────────────────────────────────────────────────
    pred = predict_fixture(home_team, away_team, poisson_model)

    # W / D / L probability cards
    p1, p2, p3 = st.columns(3)
    for col, label, prob, color in [
        (p1, f"🏠 {home_team} Win", pred["home_win_prob"], "#00c853"),
        (p2, "🤝 Draw",             pred["draw_prob"],      "#ffd600"),
        (p3, f"✈️ {away_team} Win", pred["away_win_prob"],  "#4da6ff"),
    ]:
        col.markdown(
            f"<div style='background:#1e1e1e;border-left:4px solid {color};"
            f"padding:16px 20px;border-radius:6px;text-align:center;'>"
            f"<div style='font-size:13px;color:#aaa;'>{label}</div>"
            f"<div style='font-size:36px;font-weight:700;color:{color};'>{prob:.0%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Expected goals + top scorelines
    eg1, eg2, eg3 = st.columns([2, 2, 3])
    with eg1:
        st.metric(f"{home_team} xG", f"{pred['exp_home_goals']:.2f}")
    with eg2:
        st.metric(f"{away_team} xG", f"{pred['exp_away_goals']:.2f}")
    with eg3:
        # Most likely scorelines as a compact list
        st.markdown("**Most Likely Scorelines**")
        scoreline_md = "  ".join(
            f"`{sc}` {p:.0%}" for sc, p in pred["top_scorelines"]
        )
        st.markdown(scoreline_md)

    st.markdown("---")

    # ── Team form comparison (current season grades) ──────────────────────────
    st.subheader("Form Comparison")

    # Pull the most recent season's grade row for each team
    def _latest_grade(team: str, grades_df: pd.DataFrame) -> pd.Series | None:
        rows = grades_df[grades_df["team"] == team]
        if rows.empty:
            return None
        latest_season = rows["season"].max()
        r = rows[rows["season"] == latest_season]
        return r.iloc[0] if not r.empty else None

    home_grade = _latest_grade(home_team, season_grades)
    away_grade = _latest_grade(away_team, season_grades)

    if home_grade is not None and away_grade is not None:
        grade_keys = [
            ("attack_grade",  "⚔️ Attack"),
            ("defense_grade", "🛡️ Defense"),
            ("style_grade",   "🎨 Style"),
            ("overall_grade", "⭐ Overall"),
        ]

        # Horizontal bar chart showing both teams side by side for each grade
        fig, axes = plt.subplots(1, len(grade_keys), figsize=(13, 3))

        for ax, (key, label) in zip(axes, grade_keys):
            hg = float(home_grade.get(key, 5.5))
            ag = float(away_grade.get(key, 5.5))
            teams_lbl = [home_team, away_team]
            vals      = [hg, ag]
            bar_colors = [
                "#00c853" if v >= 7 else ("#ffd600" if v >= 4.5 else "#ff1744")
                for v in vals
            ]
            bars = ax.barh(teams_lbl, vals, color=bar_colors, alpha=0.9)
            ax.set_xlim(0, 10)
            ax.axvline(5.5, color="white", linestyle="--", alpha=0.3, lw=0.8)
            ax.set_title(label, color="white", fontsize=10)
            ax.set_xlabel("Grade", color="white", fontsize=8)

            # Value labels inside bars
            for bar, val in zip(bars, vals):
                ax.text(
                    min(val - 0.3, 9.4), bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", ha="right",
                    color="white", fontsize=9, fontweight="bold",
                )
            ax.tick_params(colors="white", labelsize=8)
            ax.set_facecolor("#121212")

        fig.patch.set_facecolor("#0e1117")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Grade data not available for one or both teams.")

    # ── Model info expander ──────────────────────────────────────────────────
    with st.expander("ℹ️ How the model works"):
        st.markdown(f"""
**Poisson goals model**

Expected goals are computed for each side using team-specific attack and
defense strength ratings, relative to the league average:

```
λ_home = {poisson_model['league_avg']:.3f} × attack({home_team}) × defense({away_team}) × {poisson_model['home_advantage']:.3f}
λ_away = {poisson_model['league_avg']:.3f} × attack({away_team}) × defense({home_team})
```

Ratings use each team's last **10 matches** (xG-based) and are shrunk
toward the league average to prevent extremes from small samples.  W/D/L
probabilities are summed over a 7×7 scoreline grid.

| Metric | Value |
|---|---|
| League avg xG/team/match | {poisson_model['league_avg']:.3f} |
| Home advantage factor | {poisson_model['home_advantage']:.3f} |
| {home_team} attack | {poisson_model['team_attack'].get(home_team, 0.85):.3f} |
| {home_team} defense | {poisson_model['team_defense'].get(home_team, 0.85):.3f} |
| {away_team} attack | {poisson_model['team_attack'].get(away_team, 0.85):.3f} |
| {away_team} defense | {poisson_model['team_defense'].get(away_team, 0.85):.3f} |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# ⚡  SHOT QUALITY
# ══════════════════════════════════════════════════════════════════════════════

elif view == "⚡ Shot Quality":
    st.title("⚡ Shot Quality")
    st.markdown("Zone quality maps and team shot-quality rankings.")

    shots_all = S["shots"].copy()
    if shots_all.empty:
        st.warning("No shot data available for this season.")
        st.stop()

    needed_sh = {"team", "x", "y", "xg", "result"}
    missing_sh = needed_sh - set(shots_all.columns)
    if missing_sh:
        st.warning(f"shots data missing columns: {missing_sh}")
        st.stop()

    for col in ("x", "y", "xg"):
        shots_all[col] = pd.to_numeric(shots_all[col], errors="coerce")
    shots_all = shots_all.dropna(subset=["x", "y", "xg"])

    all_teams = team_list(shots_all, "team")
    if not all_teams:
        st.warning("No teams found in shot data.")
        st.stop()

    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        sel_team = st.selectbox("Team:", all_teams)
    with ctrl_col2:
        side_filter = st.radio("Side:", ["All", "Home Only", "Away Only"])
    with ctrl_col3:
        show_opp = st.checkbox("Include opponent shots", value=False)

    # Filter by team, then optionally by side using the `side` column that
    # shots.csv already carries ("home" / "away") — no match_summary join needed.
    team_shots = shots_all[shots_all["team"] == sel_team].copy()
    if side_filter != "All" and "side" in team_shots.columns:
        side_val = "home" if side_filter == "Home Only" else "away"
        team_shots = team_shots[team_shots["side"] == side_val]

    # shots_plot is always the team's own shots. When show_opp is requested,
    # build a separate opp_shots frame used later for the second pitch map.
    shots_plot = team_shots
    opp_shots = pd.DataFrame()
    if show_opp and "opponent" in shots_all.columns:
        opp_shots = shots_all[shots_all["opponent"] == sel_team].copy()

    if shots_plot.empty:
        st.warning(f"No shots found for {sel_team} with current filters.")
        st.stop()

    n_shots = len(shots_plot)
    avg_xg = shots_plot["xg"].mean()
    goals = shots_plot[shots_plot["result"].fillna("").str.lower() == "goal"]["xg"].count() if "result" in shots_plot.columns else 0
    total_xg = shots_plot["xg"].sum()
    xg_overperf = goals - total_xg

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Shots", n_shots)
    m2.metric("Goals", int(goals))
    m3.metric("xG Total", f"{total_xg:.2f}")
    m4.metric("Avg xG/Shot", f"{avg_xg:.3f}")
    m5.metric("xG Overperf.", f"{xg_overperf:+.2f}")

    # Half-pitch hexbin + scatter
    pitch = VerticalPitch(
        pitch_type="opta",
        half=True,
        pitch_color="#1a472a",
        line_color="#aaaaaa",
        linewidth=1.5,
    )
    fig_pitch, ax_pitch = pitch.draw(figsize=(8, 6))
    fig_pitch.patch.set_facecolor("#0e1117")
    ax_pitch.set_facecolor("#1a472a")

    # Hexbin density background
    if len(shots_plot) >= 3:
        pitch.hexbin(
            shots_plot["x"],
            shots_plot["y"],
            ax=ax_pitch,
            gridsize=10,
            cmap="YlOrRd",
            zorder=2,
            alpha=0.65,
        )

    # High-quality shots overlay (xg > 0.2)
    hq = shots_plot[shots_plot["xg"] > 0.2].copy()
    if not hq.empty:
        is_goal = hq["result"].fillna("").str.lower() == "goal" if "result" in hq.columns else pd.Series([False] * len(hq))
        hq_colors = ["gold" if g else "white" for g in is_goal]
        pitch.scatter(
            hq["x"],
            hq["y"],
            ax=ax_pitch,
            s=hq["xg"] * 600,
            c=hq_colors,
            edgecolors="#333333",
            linewidth=0.5,
            zorder=4,
            alpha=0.9,
        )

    ax_pitch.set_title(
        f"{sel_team} — Shot Quality Map ({n_shots} shots, avg xG: {avg_xg:.3f}/shot)",
        color="white",
        pad=10,
    )

    # Legend patches
    patch_goal = mpatches.Patch(color="gold", label="Goal (xG > 0.2)")
    patch_shot = mpatches.Patch(color="white", label="Shot (xG > 0.2)")
    ax_pitch.legend(handles=[patch_goal, patch_shot], loc="lower left",
                    facecolor="#1e1e1e", labelcolor="white", fontsize=8)

    st.pyplot(fig_pitch)
    plt.close(fig_pitch)

    # If opponent shots visible, also plot them
    if show_opp and "opponent" in shots_all.columns:
        if not opp_shots.empty:
            st.subheader(f"Shots Faced by {sel_team}")
            pitch_opp = VerticalPitch(
                pitch_type="opta", half=True,
                pitch_color="#1a472a", line_color="#aaaaaa", linewidth=1.5,
            )
            fig_opp, ax_opp = pitch_opp.draw(figsize=(8, 6))
            fig_opp.patch.set_facecolor("#0e1117")
            ax_opp.set_facecolor("#1a472a")
            if len(opp_shots) >= 3:
                pitch_opp.hexbin(opp_shots["x"], opp_shots["y"], ax=ax_opp,
                                 gridsize=10, cmap="Blues", zorder=2, alpha=0.65)
            hq_opp = opp_shots[opp_shots["xg"] > 0.2].copy()
            if not hq_opp.empty:
                is_goal_opp = hq_opp["result"].fillna("").str.lower() == "goal" if "result" in hq_opp.columns else pd.Series([False] * len(hq_opp))
                pitch_opp.scatter(
                    hq_opp["x"], hq_opp["y"], ax=ax_opp,
                    s=hq_opp["xg"] * 600,
                    c=["gold" if g else "#ff6b6b" for g in is_goal_opp],
                    edgecolors="#333333", linewidth=0.5, zorder=4, alpha=0.9,
                )
            n_opp = len(opp_shots)
            avg_xg_opp = opp_shots["xg"].mean()
            ax_opp.set_title(
                f"Shots Against {sel_team} ({n_opp} shots, avg xG: {avg_xg_opp:.3f}/shot)",
                color="white", pad=10,
            )
            st.pyplot(fig_opp)
            plt.close(fig_opp)

    # Team ranking chart — avg xG per shot across all teams
    st.subheader("Team Shot Quality Ranking (Avg xG/Shot)")
    rank_df = (
        shots_all.groupby("team")["xg"]
        .agg(avg_xg_shot="mean", total_shots="count")
        .reset_index()
    )
    rank_df = rank_df[rank_df["total_shots"] >= 10].sort_values("avg_xg_shot", ascending=True)

    if not rank_df.empty:
        rank_df = rank_df.copy()
        rank_df["highlight"] = rank_df["team"].apply(
            lambda t: "Selected" if t == sel_team else "Other"
        )
        fig_rank = px.bar(
            rank_df, x="avg_xg_shot", y="team",
            orientation="h",
            color="highlight",
            color_discrete_map={"Selected": GREEN, "Other": PURPLE},
            hover_name="team",
            hover_data={"avg_xg_shot": ":.4f", "total_shots": True, "highlight": False},
            labels={"avg_xg_shot": "Avg xG per Shot", "team": "",
                    "total_shots": "Total Shots", "highlight": ""},
            title="Shot Quality — All Teams",
            height=max(320, len(rank_df) * 25),
        )
        fig_rank.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"), showlegend=False,
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            title_font=dict(color="white"),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_rank, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ⚔️  HEAD-TO-HEAD
# ══════════════════════════════════════════════════════════════════════════════

elif view == "⚔️ Head-to-Head":
    st.title("⚔️ Head-to-Head")
    st.markdown("Radar comparison, historical record, and key stats.")

    ms_all_h2h    = D["match_summary"].copy()   # all seasons — for radar/stats
    crossref_all  = D["match_crossref"].copy()

    if ms_all_h2h.empty:
        st.warning("No match summary data available.")
        st.stop()

    # Season selector for H2H stats — defaults to the most recent season that
    # has ESPN match stats (possession, pass %, etc.).  2025/26 onward may only
    # have xG (Understat) until the ESPN collector is updated.
    _pct_check_col = "home_possession_pct"
    seasons_with_espn = []
    if "season" in ms_all_h2h.columns and _pct_check_col in ms_all_h2h.columns:
        seasons_with_espn = (
            ms_all_h2h[ms_all_h2h[_pct_check_col].notna() & (ms_all_h2h[_pct_check_col] != 0)]
            ["season"].dropna().unique().tolist()
        )
        seasons_with_espn = sorted(seasons_with_espn, reverse=True)

    all_seasons_h2h = sorted(
        ms_all_h2h["season"].dropna().unique().tolist(), reverse=True
    ) if "season" in ms_all_h2h.columns else []

    hc1, hc2, hc3 = st.columns([1, 1, 1])
    with hc3:
        h2h_season = st.selectbox(
            "Stats season:",
            all_seasons_h2h,
            index=0,
            key="h2h_season",
            help="Season used for radar and key stats. Pick a season with full ESPN data for possession/pass/tackle stats.",
        )

    ms = ms_all_h2h[ms_all_h2h["season"] == h2h_season].copy() if "season" in ms_all_h2h.columns else ms_all_h2h.copy()

    # Warn if selected season is missing ESPN match stats
    if _pct_check_col in ms.columns and ms[_pct_check_col].isna().all():
        st.warning(
            f"⚠️ ESPN match stats (possession, pass %, shots, tackles) are not available "
            f"for **{h2h_season}** — only xG will appear in the radar. "
            f"Select **{seasons_with_espn[0]}** for full stats."
            if seasons_with_espn else
            f"⚠️ ESPN match stats are not available for **{h2h_season}**."
        )

    all_teams_h2h = team_list(ms_all_h2h, "home_team", "away_team")
    if len(all_teams_h2h) < 2:
        st.warning("Not enough teams in match summary data.")
        st.stop()

    with hc1:
        team_a = st.selectbox("Team A:", all_teams_h2h, index=0)
    with hc2:
        default_b_idx = 1 if (team_a == all_teams_h2h[0] and len(all_teams_h2h) > 1) else 0
        team_b = st.selectbox("Team B:", all_teams_h2h, index=default_b_idx)

    if team_a == team_b:
        st.warning("Please select two different teams.")
        st.stop()

    # ── Build per-team averages from match_summary ────────────────────────────
    team_stats_h2h = (
        _flatten_ms(ms).copy()
        .groupby("team").mean(numeric_only=True).reset_index()
    )

    metric_cols = []
    metric_labels = []
    for col, label in [
        ("xg_for", "xG For/Game"),
        ("xg_against", "xGA/Game"),
        ("possession_pct", "Possession %"),
        ("pass_pct", "Pass Acc %"),
        ("shot_pct", "Shot Acc %"),
        ("tackle_pct", "Tackle Acc %"),
    ]:
        if col in team_stats_h2h.columns:
            metric_cols.append(col)
            metric_labels.append(label)

    if len(metric_cols) < 3:
        st.warning("Not enough metric columns in match_summary for radar chart.")
        st.stop()

    # Normalize 0-1 across all teams
    norm_stats = team_stats_h2h.copy()
    for col in metric_cols:
        col_min = norm_stats[col].min()
        col_max = norm_stats[col].max()
        denom = col_max - col_min
        if denom > 0:
            norm_stats[col] = (norm_stats[col] - col_min) / denom
        else:
            norm_stats[col] = 0.5
    # Invert xg_against: lower xGA normalized -> higher radar value (better defense)
    if "xg_against" in metric_cols:
        norm_stats["xg_against"] = 1 - norm_stats["xg_against"]

    def _get_vals(team_name):
        row = _team_row(norm_stats, team_name)
        if row is None:
            return [0.5] * len(metric_cols)
        return [float(row.get(c, 0.5)) for c in metric_cols]

    vals_a = _get_vals(team_a)
    vals_b = _get_vals(team_b)

    n_metrics = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    # Close the polygon
    vals_a_closed = vals_a + [vals_a[0]]
    vals_b_closed = vals_b + [vals_b[0]]
    angles_closed = angles + [angles[0]]

    # ── Section 1: Radar ──────────────────────────────────────────────────────
    st.subheader("Season Stats Radar")

    # Pull raw (un-normalized) values for hover labels
    row_a_raw = _team_row(team_stats_h2h, team_a)
    row_b_raw = _team_row(team_stats_h2h, team_b)
    raw_a = [round(float(row_a_raw[c]), 3) if row_a_raw is not None and c in row_a_raw.index else 0.0
             for c in metric_cols]
    raw_b = [round(float(row_b_raw[c]), 3) if row_b_raw is not None and c in row_b_raw.index else 0.0
             for c in metric_cols]
    # Close polygons for Plotly (repeat first element)
    theta_closed = metric_labels + [metric_labels[0]]
    raw_a_closed = raw_a + [raw_a[0]]
    raw_b_closed = raw_b + [raw_b[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_a_closed,
        theta=theta_closed,
        fill="toself",
        fillcolor="rgba(0,255,133,0.15)",
        line=dict(color="#00ff85", width=2),
        name=team_a,
        customdata=[[v] for v in raw_a_closed],
        hovertemplate=(
            f"<b>{team_a}</b><br>"
            "%{theta}: %{customdata[0]:.3f}<extra></extra>"
        ),
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_b_closed,
        theta=theta_closed,
        fill="toself",
        fillcolor="rgba(245,166,35,0.15)",
        line=dict(color="#f5a623", width=2),
        name=team_b,
        customdata=[[v] for v in raw_b_closed],
        hovertemplate=(
            f"<b>{team_b}</b><br>"
            "%{theta}: %{customdata[0]:.3f}<extra></extra>"
        ),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0e1117",
            angularaxis=dict(
                tickfont=dict(color="white", size=10),
                linecolor="#444",
                gridcolor="#333",
            ),
            radialaxis=dict(visible=False, range=[0, 1]),
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(bgcolor="#1e1e1e", font=dict(color="white")),
        title=dict(text=f"{team_a} vs {team_b}",
                   font=dict(color="white", size=13)),
        height=500,
        margin=dict(t=60, b=20),
    )
    col_radar, col_spacer = st.columns([1, 1])
    with col_radar:
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Section 2: Direct H2H record ─────────────────────────────────────────
    st.subheader("Head-to-Head Record")
    if not crossref_all.empty and all(c in crossref_all.columns for c in
                                       ["home_team", "away_team", "home_goals", "away_goals"]):
        h2h_matches = crossref_all[
            ((crossref_all["home_team"] == team_a) & (crossref_all["away_team"] == team_b)) |
            ((crossref_all["home_team"] == team_b) & (crossref_all["away_team"] == team_a))
        ].copy()
        h2h_matches = h2h_matches.dropna(subset=["home_goals", "away_goals"])

        if h2h_matches.empty:
            st.info("No completed head-to-head matches found in the data.")
        else:
            h2h_matches = h2h_matches.sort_values("match_date", ascending=False)
            h2h_matches["Date"] = pd.to_datetime(h2h_matches["match_date"], errors="coerce").dt.strftime("%d %b %Y")
            h2h_matches["Home"] = h2h_matches["home_team"]
            h2h_matches["Score"] = (
                h2h_matches["home_goals"].astype(int).astype(str) + " – " +
                h2h_matches["away_goals"].astype(int).astype(str)
            )
            h2h_matches["Away"] = h2h_matches["away_team"]
            display_cols = ["Date", "Home", "Score", "Away"]
            if "season" in h2h_matches.columns:
                h2h_matches["Season"] = h2h_matches["season"]
                display_cols.append("Season")
            st.dataframe(h2h_matches[display_cols].reset_index(drop=True),
                         use_container_width=True, hide_index=True)
    else:
        st.info("match_crossref data not available for H2H record.")

    # ── Section 3: Key stats comparison table ─────────────────────────────────
    st.subheader("Key Stats Comparison")

    row_a = _team_row(team_stats_h2h, team_a)
    row_b = _team_row(team_stats_h2h, team_b)

    # Columns stored as fractions (0–1) that should display as percentages
    _pct_cols = {"possession_pct", "pass_pct", "shot_pct", "tackle_pct"}

    comparison_rows = []
    for col, label in zip(metric_cols, metric_labels):
        def _fmt(row, c=col):
            if row is None or c not in row.index:
                return None
            v = row[c]
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            if pd.isna(v):
                return None
            # possession_pct is 0-100; pass/shot/tackle_pct are 0-1 fractions
            if c in _pct_cols and c != "possession_pct":
                v = round(v * 100, 1)
            else:
                v = round(v, 3)
            return v
        val_a = _fmt(row_a)
        val_b = _fmt(row_b)
        comparison_rows.append({"Metric": label, team_a: val_a, team_b: val_b})

    comp_df = pd.DataFrame(comparison_rows)

    def highlight_better(row):
        val_a_r, val_b_r = row[team_a], row[team_b]
        if val_a_r is None or val_b_r is None or val_a_r == val_b_r:
            return [""] * len(row)
        lower_better = "xGA" in row["Metric"]
        a_wins = (val_a_r < val_b_r) if lower_better else (val_a_r > val_b_r)
        win_color = "background-color: #1a3a2a"
        result = [""] * len(row)
        a_idx = comp_df.columns.get_loc(team_a)
        b_idx = comp_df.columns.get_loc(team_b)
        result[a_idx if a_wins else b_idx] = win_color
        return result

    styled_comp = comp_df.style.apply(highlight_better, axis=1)
    st.dataframe(styled_comp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🗺️  TACTICS & FORMATION
# ══════════════════════════════════════════════════════════════════════════════
# Three tabs built entirely from WhoScored events + ESPN lineups:
#
#   Average Positions — each player's mean event coordinate, coloured by
#     position group, giving a natural picture of the team's shape.
#
#   Pass Network — nodes at average positions, edges between players who
#     passed sequentially (approximates recipient without related_player_id).
#     Edge weight = connection frequency; node size = pass volume.
#
#   Defensive Shape — scatter of tackles, interceptions, clearances and
#     recoveries; average defensive action line; breakdown by pitch third.

elif view == "🗺️ Tactics":
    st.title("🗺️ Tactics & Formation")
    st.markdown("Average positions, pass networks, and defensive shape — all from event data.")

    lineups_tac = D["lineups"].copy()
    ms_tac      = S["match_summary"].copy()

    if ms_tac.empty:
        st.warning("No match data for this season.")
        st.stop()

    all_teams_tac = team_list(ms_tac, "home_team", "away_team")
    if not all_teams_tac:
        st.warning("No teams found.")
        st.stop()

    # ── Top-level filters ─────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([1, 2, 1])
    with fc1:
        sel_tac_team = st.selectbox("Team:", all_teams_tac, key="tac_team")
    with fc3:
        sel_tac_period = st.selectbox(
            "Period:", ["Both Halves", "First Half", "Second Half"], key="tac_period"
        )

    with st.spinner("Loading events…"):
        tac_events = load_events(sel_season)

    if tac_events.empty:
        st.warning("No event data for this season.")
        st.stop()

    # Filter to selected team; build match dropdown
    team_ev = tac_events[tac_events["team"] == sel_tac_team].copy()
    if team_ev.empty:
        st.warning(f"No events found for {sel_tac_team} this season.")
        st.stop()

    team_ev["_label"] = team_ev.apply(match_label, axis=1)
    match_opts = ["📅 Season Average"] + (
        team_ev.drop_duplicates("_label")
        .sort_values("match_date", ascending=False)["_label"]
        .tolist()
    )
    with fc2:
        sel_tac_match = st.selectbox("Match:", match_opts, key="tac_match")

    # Apply match + period filters
    if sel_tac_match == "📅 Season Average":
        ev = team_ev.copy()
        match_title = f"{sel_tac_team} — {sel_season} Season Average"
    else:
        ev = team_ev[team_ev["_label"] == sel_tac_match].copy()
        match_title = f"{sel_tac_team} — {sel_tac_match}"

    if sel_tac_period == "First Half":
        ev = ev[ev["period"] == "FirstHalf"]
    elif sel_tac_period == "Second Half":
        ev = ev[ev["period"] == "SecondHalf"]

    if ev.empty:
        st.warning("No events found for this selection.")
        st.stop()

    # ── Starting lineup (single match only) ───────────────────────────────────
    _pos_map = {"Goalkeeper": "GK", "Defender": "DEF",
                "Midfielder": "MID", "Forward": "FWD"}
    starters = pd.DataFrame()
    if not lineups_tac.empty and sel_tac_match != "📅 Season Average":
        md_val = ev["match_date"].dropna().iloc[0] if not ev["match_date"].dropna().empty else None
        if md_val is not None:
            lu_m = lineups_tac[
                (lineups_tac["team"] == sel_tac_team) &
                (pd.to_datetime(lineups_tac["match_date"], errors="coerce").dt.date
                 == pd.to_datetime(md_val, errors="coerce").date())
            ]
            if not lu_m.empty:
                starters = lu_m[
                    lu_m["formation_place"].between(1, 11) &
                    lu_m["sub_in"].isna()
                ].copy()
                starters["pos_group"] = starters["position"].map(_pos_map).fillna("MID")

    # Infer formation string from position counts
    formation_label = ""
    if not starters.empty:
        n_def = (starters["pos_group"] == "DEF").sum()
        n_mid = (starters["pos_group"] == "MID").sum()
        n_fwd = (starters["pos_group"] == "FWD").sum()
        if n_def + n_mid + n_fwd >= 8:
            formation_label = f"{n_def}-{n_mid}-{n_fwd}"

    # ── Average positions per player ──────────────────────────────────────────
    avg_pos = (
        ev.dropna(subset=["x", "y", "player"])
        .groupby("player")
        .agg(avg_x=("x", "mean"), avg_y=("y", "mean"), n_events=("type", "count"))
        .reset_index()
    )
    # Attach position group (from starters if available, else unknown)
    if not starters.empty:
        avg_pos = avg_pos.merge(
            starters[["player", "pos_group"]].drop_duplicates("player"),
            on="player", how="left",
        )
    if "pos_group" not in avg_pos.columns:
        avg_pos["pos_group"] = "MID"
    avg_pos["pos_group"] = avg_pos["pos_group"].fillna("MID")

    # Colour scheme (consistent across all three tabs)
    pos_colors = {"GK": "#f1c40f", "DEF": "#3498db", "MID": "#2ecc71", "FWD": "#e74c3c"}

    tab_pos, tab_net, tab_def, tab_form, tab_sp = st.tabs([
        "📍 Average Positions",
        "🔗 Pass Network",
        "🛡️ Defensive Shape",
        "🏟️ Formation",
        "⚽ Set Pieces",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # Tab 1 — Average Positions
    # ══════════════════════════════════════════════════════════════════════
    with tab_pos:
        st.subheader(match_title)
        if formation_label:
            st.caption(f"Inferred formation: **{formation_label}**  ·  based on starting lineup positions")

        # Summary metrics
        avg_x_team = avg_pos["avg_x"].mean()
        def_row = avg_pos[avg_pos["pos_group"].isin(["GK", "DEF"])]
        fwd_row = avg_pos[avg_pos["pos_group"] == "FWD"]
        avg_def_line = def_row["avg_x"].mean() if not def_row.empty else np.nan
        avg_att_line = fwd_row["avg_x"].mean() if not fwd_row.empty else np.nan

        mp1, mp2, mp3, mp4 = st.columns(4)
        mp1.metric("Players tracked", len(avg_pos))
        mp2.metric("Team avg position", f"{avg_x_team:.0f}%" if not np.isnan(avg_x_team) else "—")
        mp3.metric("Defensive line", f"{avg_def_line:.0f}%" if not np.isnan(avg_def_line) else "—")
        mp4.metric("Attack height", f"{avg_att_line:.0f}%" if not np.isnan(avg_att_line) else "—")

        fig_pos, ax_pos = plt.subplots(figsize=(12, 7))
        pitch_pos = Pitch(pitch_type="opta", pitch_color=PITCH_GREEN,
                          line_color="white", line_zorder=2)
        pitch_pos.draw(ax=ax_pos)

        for _, r in avg_pos.iterrows():
            col  = pos_colors.get(r["pos_group"], "#aaaaaa")
            size = float(np.clip(r["n_events"] / 5, 80, 600))
            ax_pos.scatter(r["avg_x"], r["avg_y"], c=col, s=size,
                           zorder=5, alpha=0.92, edgecolors="white", linewidths=1.2)
            parts = str(r["player"]).split()
            lbl   = parts[-1] if len(parts) > 1 else parts[0]
            ax_pos.text(r["avg_x"], r["avg_y"] + 3.5, lbl,
                        ha="center", va="bottom", fontsize=7,
                        color="white", fontweight="bold", zorder=6)

        if not np.isnan(avg_x_team):
            ax_pos.axvline(avg_x_team, color="white", lw=0.8,
                           linestyle="--", alpha=0.4, zorder=3)

        legend_els = [
            mpatches.Patch(color=c, label=pg)
            for pg, c in pos_colors.items()
            if pg in avg_pos["pos_group"].values
        ]
        ax_pos.legend(handles=legend_els, loc="upper left",
                      facecolor="#1a1a1a", labelcolor="white", fontsize=8)
        ax_pos.set_title(f"Average Positions — {match_title}",
                         color="white", fontsize=11, pad=10)
        dark_fig_style(fig_pos, ax_pos)
        plt.tight_layout()
        st.pyplot(fig_pos)
        plt.close(fig_pos)
        st.caption(
            "Dot size = event volume (more active = larger).  "
            "Position = average of all recorded event coordinates.  "
            "Dashed line = team average x-position."
        )

    # ══════════════════════════════════════════════════════════════════════
    # Tab 2 — Pass Network
    # ══════════════════════════════════════════════════════════════════════
    with tab_net:
        st.subheader(match_title)

        passes = (
            ev[ev["type"] == "Pass"]
            .dropna(subset=["player", "x", "y"])
            .sort_values("minute")
            .reset_index(drop=True)
        )

        if len(passes) < 5:
            st.info("Not enough pass data for this selection.")
        else:
            total_passes = len(passes)
            succ_passes  = (passes["outcome_type"].fillna("").str.lower() == "successful").sum()
            pass_comp    = 100 * succ_passes / total_passes if total_passes else 0
            avg_pass_len = np.sqrt(
                (passes["end_x"].fillna(passes["x"]) - passes["x"]) ** 2 +
                (passes["end_y"].fillna(passes["y"]) - passes["y"]) ** 2
            ).mean()

            pn1, pn2, pn3 = st.columns(3)
            pn1.metric("Total passes", total_passes)
            pn2.metric("Completion", f"{pass_comp:.0f}%")
            pn3.metric("Avg length", f"{avg_pass_len:.1f}")

            # Sequential connection approximation:
            # Each pass is linked to the next pass by the same team within ≤ 3 min.
            passes["_next_player"] = passes["player"].shift(-1)
            passes["_next_min"]    = passes["minute"].shift(-1)
            conn = passes[
                (passes["_next_min"] - passes["minute"]).fillna(999) <= 3
            ][["player", "_next_player"]].copy()
            conn.columns = ["from_player", "to_player"]
            conn = conn[conn["from_player"] != conn["to_player"]]

            conn_counts = (
                conn.groupby(["from_player", "to_player"])
                .size().reset_index(name="count")
            )
            conn_counts = conn_counts[conn_counts["count"] >= 2]

            # Node positions + pass volume
            pass_vol = passes.groupby("player").size().reset_index(name="pass_count")
            net_pos  = avg_pos.merge(pass_vol, on="player", how="left")
            net_pos["pass_count"] = net_pos["pass_count"].fillna(0)

            fig_net, ax_net = plt.subplots(figsize=(12, 7))
            pitch_net = Pitch(pitch_type="opta", pitch_color=PITCH_GREEN,
                              line_color="white", line_zorder=2)
            pitch_net.draw(ax=ax_net)

            # Draw edges first (behind nodes)
            if not conn_counts.empty:
                max_c = conn_counts["count"].max()
                for _, edge in conn_counts.iterrows():
                    fp = net_pos[net_pos["player"] == edge["from_player"]]
                    tp = net_pos[net_pos["player"] == edge["to_player"]]
                    if fp.empty or tp.empty:
                        continue
                    lw    = 1.0 + 5.0 * (edge["count"] / max_c)
                    alpha = 0.25 + 0.55 * (edge["count"] / max_c)
                    ax_net.plot(
                        [fp["avg_x"].values[0], tp["avg_x"].values[0]],
                        [fp["avg_y"].values[0], tp["avg_y"].values[0]],
                        color="#00ff85", linewidth=lw, alpha=alpha, zorder=3,
                    )

            # Draw nodes on top
            for _, r in net_pos.iterrows():
                col  = pos_colors.get(r.get("pos_group", "MID"), "#aaaaaa")
                size = float(np.clip(r["pass_count"] * 7, 80, 700))
                ax_net.scatter(r["avg_x"], r["avg_y"], c=col, s=size,
                               zorder=5, alpha=0.95,
                               edgecolors="white", linewidths=1.5)
                parts = str(r["player"]).split()
                lbl   = parts[-1] if len(parts) > 1 else parts[0]
                ax_net.text(r["avg_x"], r["avg_y"] + 3.5, lbl,
                            ha="center", va="bottom", fontsize=7,
                            color="white", fontweight="bold", zorder=6)

            ax_net.set_title(f"Pass Network — {match_title}",
                             color="white", fontsize=11, pad=10)
            dark_fig_style(fig_net, ax_net)
            plt.tight_layout()
            st.pyplot(fig_net)
            plt.close(fig_net)
            st.caption(
                "Node position = average event position.  "
                "Node size = pass volume.  "
                "Line thickness = connection frequency (≥ 2 sequential passes).  "
                "Connections are approximated from consecutive team passes within 3 minutes."
            )

            if not conn_counts.empty:
                st.markdown("**Strongest connections**")
                st.dataframe(
                    conn_counts.sort_values("count", ascending=False)
                    .head(10)
                    .rename(columns={"from_player": "From", "to_player": "To", "count": "Passes"}),
                    use_container_width=True, hide_index=True,
                )

    # ══════════════════════════════════════════════════════════════════════
    # Tab 3 — Defensive Shape
    # ══════════════════════════════════════════════════════════════════════
    with tab_def:
        st.subheader(match_title)

        def_type_colors = {
            "Tackle":       "#e74c3c",
            "Interception": "#3498db",
            "Clearance":    "#f39c12",
            "BallRecovery": "#2ecc71",
            "BlockedShot":  "#9b59b6",
        }
        def_ev = (
            ev[ev["type"].isin(def_type_colors.keys())]
            .dropna(subset=["x", "y"])
            .copy()
        )

        if def_ev.empty:
            st.info("No defensive action data found for this selection.")
        else:
            total_def    = len(def_ev)
            n_att_third  = (def_ev["x"] >= 67).sum()
            n_mid_third  = ((def_ev["x"] >= 33) & (def_ev["x"] < 67)).sum()
            n_def_third  = (def_ev["x"] < 33).sum()
            avg_press_x  = def_ev["x"].mean()
            press_pct    = 100 * n_att_third / total_def if total_def else 0

            dd1, dd2, dd3, dd4 = st.columns(4)
            dd1.metric("Defensive actions", total_def)
            dd2.metric("Press height",      f"{avg_press_x:.0f}%")
            dd3.metric("In attacking ⅓",   f"{press_pct:.0f}%")
            dd4.metric("In defensive ⅓",   f"{100*n_def_third/total_def:.0f}%")

            c_pitch, c_bar = st.columns([3, 1])

            with c_pitch:
                fig_def, ax_def = plt.subplots(figsize=(10, 6))
                pitch_def = Pitch(pitch_type="opta", pitch_color=PITCH_GREEN,
                                  line_color="white", line_zorder=2)
                pitch_def.draw(ax=ax_def)

                for dtype, dcolor in def_type_colors.items():
                    sub = def_ev[def_ev["type"] == dtype]
                    if not sub.empty:
                        ax_def.scatter(sub["x"], sub["y"], c=dcolor,
                                       s=28, alpha=0.65, zorder=4,
                                       label=dtype, edgecolors="none")

                # Average defensive action x-line
                ax_def.axvline(avg_press_x, color="white", lw=1.5,
                               linestyle="--", alpha=0.75, zorder=5,
                               label=f"Avg line ({avg_press_x:.0f}%)")

                # Light shading for pitch thirds
                ax_def.axvspan(0,   33,  alpha=0.04, color="#3498db")
                ax_def.axvspan(67,  100, alpha=0.04, color="#e74c3c")

                ax_def.legend(loc="upper left", facecolor="#1a1a1a",
                              labelcolor="white", fontsize=7, markerscale=1.3)
                ax_def.set_title(f"Defensive Shape — {match_title}",
                                 color="white", fontsize=11, pad=10)
                dark_fig_style(fig_def, ax_def)
                plt.tight_layout()
                st.pyplot(fig_def)
                plt.close(fig_def)

            with c_bar:
                st.markdown("**By pitch third**")
                thirds_df = pd.DataFrame({
                    "Third":   ["Attacking ⅓", "Middle ⅓", "Defensive ⅓"],
                    "Actions": [int(n_att_third), int(n_mid_third), int(n_def_third)],
                })
                thirds_df["Pct"] = (
                    100 * thirds_df["Actions"] / total_def
                ).round(1).astype(str) + "%"
                fig_thirds = px.bar(
                    thirds_df, x="Actions", y="Third",
                    orientation="h",
                    color="Third",
                    color_discrete_map={
                        "Attacking ⅓": "#e74c3c",
                        "Middle ⅓":    "#f39c12",
                        "Defensive ⅓": "#3498db",
                    },
                    text="Pct",
                    hover_data={"Pct": True, "Actions": True, "Third": False},
                    height=220,
                )
                fig_thirds.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="white"), showlegend=False,
                    xaxis=dict(gridcolor="#333"),
                    yaxis=dict(gridcolor="#333"),
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_thirds, use_container_width=True)

            # Action type breakdown table
            st.markdown("**By action type**")
            type_summary = (
                def_ev.groupby("type")
                .agg(
                    Total=("type", "count"),
                    Successful=("outcome_type", lambda x:
                                (x.fillna("").str.lower() == "successful").sum()),
                )
                .reset_index()
            )
            type_summary["Success %"] = (
                100 * type_summary["Successful"] / type_summary["Total"]
            ).round(1)
            st.dataframe(
                type_summary.rename(columns={"type": "Action"})
                .sort_values("Total", ascending=False),
                use_container_width=True, hide_index=True,
            )

    # ══════════════════════════════════════════════════════════════════════
    # Tab 4 — Formation
    # ══════════════════════════════════════════════════════════════════════
    with tab_form:
        st.subheader(match_title)

        formations_tac = D.get("formations", pd.DataFrame())

        # ── Resolve formation for the selected match / season average ─────────
        team_forms = pd.DataFrame()
        if not formations_tac.empty and "team" in formations_tac.columns:
            team_forms = formations_tac[formations_tac["team"] == sel_tac_team].copy()
            if "season" in team_forms.columns:
                team_forms = team_forms[team_forms["season"] == sel_season]

        # Current selection
        if sel_tac_match != "📅 Season Average" and not team_forms.empty:
            # Match the selected match date to pick the formation
            md_val = ev["match_date"].dropna().iloc[0] if not ev["match_date"].dropna().empty else None
            match_form = pd.DataFrame()
            if md_val is not None:
                match_form = team_forms[
                    pd.to_datetime(team_forms["match_date"], errors="coerce").dt.date
                    == pd.to_datetime(md_val, errors="coerce").date()
                ]
            if not match_form.empty:
                formation_str = match_form["formation"].iloc[0]
                st.markdown(
                    f"<h1 style='text-align:center;font-size:3rem;letter-spacing:0.15em;"
                    f"color:#00ff85'>{formation_str}</h1>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='text-align:center;color:#aaa'>"
                    f"{sel_tac_team} vs {match_title.split('—')[-1].strip()} — "
                    f"Formation from WhoScored data</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No WhoScored formation record for this match.")
        elif sel_tac_match == "📅 Season Average" and not team_forms.empty:
            st.info("Season-average mode shows formation history across all matches.")

        # ── Season formation history table ────────────────────────────────────
        if not team_forms.empty and "formation" in team_forms.columns:
            st.markdown("### Formation history this season")

            form_hist = team_forms.copy()
            if "match_date" in form_hist.columns:
                form_hist["match_date"] = pd.to_datetime(
                    form_hist["match_date"], errors="coerce"
                )
                form_hist = form_hist.sort_values("match_date", ascending=False)

            # Opponent column
            if "home_team" in form_hist.columns and "away_team" in form_hist.columns:
                form_hist["Opponent"] = form_hist.apply(
                    lambda r: r["away_team"] if r["home_team"] == sel_tac_team
                    else r["home_team"], axis=1
                )
                form_hist["H/A"] = form_hist.apply(
                    lambda r: "H" if r["home_team"] == sel_tac_team else "A", axis=1
                )

            display_cols = {}
            if "match_date" in form_hist.columns:
                form_hist["Date"] = form_hist["match_date"].dt.strftime("%d %b %Y")
                display_cols["Date"] = "Date"
            if "Opponent" in form_hist.columns:
                display_cols["Opponent"] = "Opponent"
            if "H/A" in form_hist.columns:
                display_cols["H/A"] = "H/A"
            display_cols["formation"] = "Formation"

            st.dataframe(
                form_hist[[c for c in display_cols if c in form_hist.columns]]
                .rename(columns={"formation": "Formation"}),
                use_container_width=True, hide_index=True,
            )

            # Formation frequency bar chart
            # Use explicit column assignment — avoids pandas version differences
            # in value_counts().reset_index() column naming (pre/post 2.0).
            _vc = team_forms["formation"].value_counts().reset_index()
            _vc.columns = ["Formation", "Matches"]
            form_counts = _vc

            if len(form_counts) > 1:
                st.markdown("### Formation usage")
                fig_fc = px.bar(
                    form_counts, x="Formation", y="Matches",
                    color="Formation",
                    text="Matches",
                    title=f"{sel_tac_team} — {sel_season} formation usage",
                    height=280,
                )
                fig_fc.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="white"), showlegend=False,
                    xaxis=dict(gridcolor="#333"),
                    yaxis=dict(gridcolor="#333"),
                    margin=dict(t=40, b=10, l=10, r=10),
                )
                fig_fc.update_traces(
                    hovertemplate="<b>%{x}</b><br>Matches: %{y}<extra></extra>",
                    textposition="outside",
                )
                st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info(
                "Formation data comes from WhoScored events. "
                "Rebuild the database (`python build_db.py`) to populate it."
            )

    # ══════════════════════════════════════════════════════════════════════
    # Tab 5 — Set Pieces
    # ══════════════════════════════════════════════════════════════════════
    with tab_sp:
        st.subheader(match_title)

        # Split into corners vs free kicks
        sp_corners = ev[
            (ev.get("is_corner", pd.Series(0, index=ev.index)) == 1) |
            (ev["type"] == "CornerAwarded")
        ].copy() if "is_corner" in ev.columns else pd.DataFrame()

        # Use is_corner flag if available (proper deliveries), else warn
        if "is_corner" not in ev.columns:
            st.warning(
                "Set-piece flags not found in this database. "
                "Re-run `python build_db.py` to add corner/free-kick flags."
            )
        else:
            corners_ev  = ev[(ev["is_corner"]   == 1)].copy()
            freekick_ev = ev[(ev["is_freekick"]  == 1)].copy()

            sp_tab1, sp_tab2 = st.tabs(["🚩 Corners", "🎯 Free Kicks"])

            # ── Corners ───────────────────────────────────────────────────────
            with sp_tab1:
                if corners_ev.empty:
                    st.info("No corner deliveries for this selection.")
                else:
                    corners_ev = corners_ev.dropna(subset=["x", "y", "end_x", "end_y"])
                    total_c    = len(corners_ev)
                    succ_c     = (corners_ev["outcome_type"].fillna("").str.lower() == "successful").sum()
                    # Corners from the left channel (y < 50 = bottom of pitch → right side
                    # in opta coords) vs right channel (y > 50)
                    left_c  = (corners_ev["y"] < 10).sum()   # near-left corner flag
                    right_c = (corners_ev["y"] > 90).sum()   # near-right corner flag

                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("Total corners", total_c)
                    cm2.metric("Successful delivery", f"{succ_c}")
                    cm3.metric("Completion %", f"{100*succ_c/total_c:.0f}%" if total_c else "—")
                    cm4.metric("Left / Right", f"{left_c} / {right_c}")

                    fig_c, ax_c = plt.subplots(figsize=(12, 7))
                    pitch_c = Pitch(pitch_type="opta", pitch_color=PITCH_GREEN,
                                    line_color="white", line_zorder=2)
                    pitch_c.draw(ax=ax_c)

                    # Draw delivery arrows: origin (corner flag) → landing zone
                    for _, row in corners_ev.iterrows():
                        succ = str(row.get("outcome_type", "")).lower() == "successful"
                        color = "#00ff85" if succ else "#e74c3c"
                        alpha = 0.55 if succ else 0.35
                        ax_c.annotate(
                            "", xy=(row["end_x"], row["end_y"]),
                            xytext=(row["x"], row["y"]),
                            arrowprops=dict(
                                arrowstyle="->", color=color,
                                lw=1.1, alpha=alpha,
                            ),
                            zorder=4,
                        )

                    # Shade penalty area as target zone
                    ax_c.add_patch(plt.Rectangle(
                        (83, 21.1), 17, 57.8, linewidth=0,
                        edgecolor="none", facecolor="#ffffff", alpha=0.05, zorder=2,
                    ))

                    ax_c.set_title(f"Corner Deliveries — {match_title}",
                                   color="white", fontsize=11, pad=10)
                    from matplotlib.lines import Line2D
                    legend_els_c = [
                        Line2D([0], [0], color="#00ff85", lw=2, label="Successful"),
                        Line2D([0], [0], color="#e74c3c", lw=2, label="Unsuccessful"),
                    ]
                    ax_c.legend(handles=legend_els_c, loc="upper left",
                                facecolor="#1a1a1a", labelcolor="white", fontsize=8)
                    dark_fig_style(fig_c, ax_c)
                    plt.tight_layout()
                    st.pyplot(fig_c)
                    plt.close(fig_c)
                    st.caption(
                        "Green arrows = successful delivery · Red = unsuccessful. "
                        "Shaded box = penalty area. "
                        "Arrow origin = corner flag position."
                    )

                    # Landing zone breakdown (thirds of goal area width)
                    if total_c >= 3:
                        st.markdown("**Delivery zones (end position)**")
                        near_post  = ((corners_ev["end_y"] < 37) | (corners_ev["end_y"] > 63)).sum()
                        six_yard   = ((corners_ev["end_x"] > 94) & corners_ev["end_y"].between(21, 79)).sum()
                        penalty    = ((corners_ev["end_x"].between(83, 94)) & corners_ev["end_y"].between(21, 79)).sum()
                        zone_df = pd.DataFrame({
                            "Zone":    ["Near/Far Post", "6-yard box", "Penalty area", "Other"],
                            "Corners": [
                                int(near_post), int(six_yard), int(penalty),
                                int(total_c - near_post - six_yard - penalty),
                            ],
                        })
                        zone_df["Corners"] = zone_df["Corners"].clip(lower=0)
                        fig_cz = px.bar(
                            zone_df, x="Corners", y="Zone", orientation="h",
                            color="Zone",
                            text="Corners",
                            height=200,
                        )
                        fig_cz.update_layout(
                            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font=dict(color="white"), showlegend=False,
                            xaxis=dict(gridcolor="#333"),
                            yaxis=dict(gridcolor="#333"),
                            margin=dict(t=10, b=10, l=10, r=10),
                        )
                        fig_cz.update_traces(
                            hovertemplate="<b>%{y}</b><br>Corners: %{x}<extra></extra>",
                        )
                        st.plotly_chart(fig_cz, use_container_width=True)

            # ── Free Kicks ────────────────────────────────────────────────────
            with sp_tab2:
                if freekick_ev.empty:
                    st.info("No free kick deliveries for this selection.")
                else:
                    freekick_ev = freekick_ev.dropna(subset=["x", "y", "end_x", "end_y"])
                    total_fk    = len(freekick_ev)
                    succ_fk     = (freekick_ev["outcome_type"].fillna("").str.lower() == "successful").sum()
                    att_fk      = (freekick_ev["x"] >= 60).sum()   # attacking half FKs
                    def_fk      = (freekick_ev["x"] < 40).sum()    # defensive FKs

                    fm1, fm2, fm3, fm4 = st.columns(4)
                    fm1.metric("Total FK deliveries", total_fk)
                    fm2.metric("Completion %", f"{100*succ_fk/total_fk:.0f}%" if total_fk else "—")
                    fm3.metric("In attacking half", att_fk)
                    fm4.metric("In defensive half", def_fk)

                    # Zone filter
                    fk_zone = st.radio(
                        "Show:", ["All", "Attacking half (x ≥ 60)", "Defensive half (x < 40)"],
                        horizontal=True, key="fk_zone",
                    )
                    if fk_zone == "Attacking half (x ≥ 60)":
                        fk_plot = freekick_ev[freekick_ev["x"] >= 60]
                    elif fk_zone == "Defensive half (x < 40)":
                        fk_plot = freekick_ev[freekick_ev["x"] < 40]
                    else:
                        fk_plot = freekick_ev

                    fig_fk, ax_fk = plt.subplots(figsize=(12, 7))
                    pitch_fk = Pitch(pitch_type="opta", pitch_color=PITCH_GREEN,
                                     line_color="white", line_zorder=2)
                    pitch_fk.draw(ax=ax_fk)

                    for _, row in fk_plot.iterrows():
                        succ  = str(row.get("outcome_type", "")).lower() == "successful"
                        color = "#00ff85" if succ else "#e74c3c"
                        alpha = 0.55 if succ else 0.35
                        ax_fk.annotate(
                            "", xy=(row["end_x"], row["end_y"]),
                            xytext=(row["x"], row["y"]),
                            arrowprops=dict(
                                arrowstyle="->", color=color,
                                lw=1.0, alpha=alpha,
                            ),
                            zorder=4,
                        )
                        # Mark origin point
                        ax_fk.scatter(row["x"], row["y"], c=color,
                                      s=18, alpha=0.7, zorder=5, edgecolors="none")

                    # Shade the danger zone (penalty area, attacking end)
                    ax_fk.add_patch(plt.Rectangle(
                        (83, 21.1), 17, 57.8, linewidth=0,
                        edgecolor="none", facecolor="#ffffff", alpha=0.05, zorder=2,
                    ))

                    ax_fk.set_title(f"Free Kick Deliveries — {match_title}",
                                    color="white", fontsize=11, pad=10)
                    ax_fk.legend(handles=legend_els_c, loc="upper left",
                                 facecolor="#1a1a1a", labelcolor="white", fontsize=8)
                    dark_fig_style(fig_fk, ax_fk)
                    plt.tight_layout()
                    st.pyplot(fig_fk)
                    plt.close(fig_fk)
                    st.caption(
                        "Green arrows = successful delivery · Red = unsuccessful. "
                        "Dot = FK origin. Arrow = delivery direction. "
                        "Shaded box = penalty area."
                    )

                    # FK origin heatmap by pitch zone
                    if total_fk >= 5:
                        st.markdown("**Free kick origin zones**")
                        fk_att3  = (freekick_ev["x"] >= 67).sum()
                        fk_mid3  = ((freekick_ev["x"] >= 33) & (freekick_ev["x"] < 67)).sum()
                        fk_def3  = (freekick_ev["x"] < 33).sum()
                        fk_zones_df = pd.DataFrame({
                            "Zone":      ["Attacking ⅓", "Middle ⅓", "Defensive ⅓"],
                            "Free Kicks": [int(fk_att3), int(fk_mid3), int(fk_def3)],
                        })
                        fig_fkz = px.bar(
                            fk_zones_df, x="Free Kicks", y="Zone", orientation="h",
                            color="Zone",
                            color_discrete_map={
                                "Attacking ⅓": "#e74c3c",
                                "Middle ⅓":    "#f39c12",
                                "Defensive ⅓": "#3498db",
                            },
                            text="Free Kicks",
                            height=200,
                        )
                        fig_fkz.update_layout(
                            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font=dict(color="white"), showlegend=False,
                            xaxis=dict(gridcolor="#333"),
                            yaxis=dict(gridcolor="#333"),
                            margin=dict(t=10, b=10, l=10, r=10),
                        )
                        fig_fkz.update_traces(
                            hovertemplate="<b>%{y}</b><br>Free Kicks: %{x}<extra></extra>",
                        )
                        st.plotly_chart(fig_fkz, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources**")
st.sidebar.markdown("- Understat · ESPN · WhoScored")
st.sidebar.markdown("---")
st.sidebar.info("💡 Run `python build_processed.py` after each collection cycle to refresh.")
