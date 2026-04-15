"""
dashboard.py
------------
Premier League Analytics Dashboard
Data: Understat · ESPN · WhoScored  →  data/processed/

Run with:
    streamlit run dashboard.py

Views available:
    🎯 Shot Maps      — shot locations and xG per team / player / match
    🔥 Event Heatmaps — WhoScored event density maps by type and period
    📊 Match Analysis — per-match scorecard, xG bar, forecast, shot map
    👤 Player Stats   — season stats table with xG and xA scatter plots
    📈 xG Analysis    — team-level xG aggregates and overperformance charts
"""

import io
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from mplsoccer import Pitch, VerticalPitch

# Local modules
sys.path.insert(0, "src")
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

# Inject a small CSS block to set the dark background and style metric cards.
# unsafe_allow_html=True is required to pass raw HTML/CSS into Streamlit.
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #e8e8e8; }
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Shared colour constants used across all charts so every view stays consistent.
PURPLE      = "#37003c"   # Premier League deep purple (bars, accents)
GREEN       = "#00ff85"   # Premier League neon green (highlights, goals)
PITCH_GREEN = "#1a472a"   # Dark grass green used as the pitch background

# Root folder where build_processed.py writes its output CSVs.
PROCESSED = "data/processed"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
# @st.cache_data tells Streamlit to run this function only once and cache the
# result in memory. Subsequent page interactions reuse the cached DataFrames
# instead of re-reading from disk — keeps the app fast.

@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    """Load all processed CSVs into a single dictionary of DataFrames."""

    def safe_read(name: str) -> pd.DataFrame:
        # Returns an empty DataFrame instead of crashing if a file is missing.
        # This lets the app start and show a warning rather than erroring out.
        try:
            return pd.read_csv(f"{PROCESSED}/{name}", low_memory=False)
        except FileNotFoundError:
            return pd.DataFrame()

    shots         = safe_read("shots.csv")         # Understat shot-level data
    events        = safe_read("events.csv")         # WhoScored full event stream
    match_summary = safe_read("match_summary.csv")  # One row per match (xG, goals, forecasts)
    match_crossref = safe_read("match_crossref.csv") # All fixtures incl. upcoming (NaN goals)
    player_season = safe_read("player_season.csv")  # Understat player season totals
    lineups       = safe_read("lineups.csv")         # ESPN lineup data (unused in current views)

    # Parse match_date to a proper datetime once at load time.
    # errors="coerce" turns unparseable values into NaT rather than crashing.
    for df in (shots, events, match_summary, lineups, match_crossref):
        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    return dict(
        shots=shots,
        events=events,
        match_summary=match_summary,
        match_crossref=match_crossref,
        player_season=player_season,
        lineups=lineups,
    )


# Show a spinner while the CSVs load, then store everything in D.
with st.spinner("Loading data…"):
    D = load_data()

# If every DataFrame came back empty (files not yet generated), stop early
# and tell the user what to run.
if all(v.empty for v in D.values()):
    st.error("No processed data found in `data/processed/`. "
             "Run `python build_processed.py` first.")
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
def _get_player_grades(ps_df: pd.DataFrame) -> pd.DataFrame:
    """Compute player grades from player_season.csv, cached per DataFrame."""
    if ps_df.empty:
        return pd.DataFrame()
    return compute_player_grades(ps_df)


@st.cache_data
def _get_poisson_model(crossref_df: pd.DataFrame) -> dict:
    """Build Poisson model from full crossref (all seasons), cached."""
    return build_poisson_model(crossref_df, n_recent=10)


@st.cache_data
def _get_upcoming(crossref_df: pd.DataFrame) -> pd.DataFrame:
    """Return upcoming (unplayed) fixtures from crossref, cached."""
    return get_upcoming_fixtures(crossref_df)


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
    "🎯 Shot Maps",
    "🔥 Event Heatmaps",
    "📊 Match Analysis",
    "👤 Player Stats",
    "📈 xG Analysis",
    "🏅 Team Grades",
    "⭐ Player Grades",
    "🔮 Match Predictor",
])

st.sidebar.markdown("---")


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

if view == "🎯 Shot Maps":
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

    events = S["events"]
    if events.empty:
        st.warning("No event data — run the WhoScored collector.")
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

    # Filter the full events table down to just this team before building the
    # match dropdown — otherwise match_label() would show every match in the dataset.
    team_events_all = events[events["team"] == sel_team].copy()

    with r1b:
        if {"match_date", "home_team", "away_team"}.issubset(team_events_all.columns):
            team_events_all["match_date"] = pd.to_datetime(
                team_events_all["match_date"], errors="coerce"
            )
            # Build one label per unique match, sorted most recent first.
            # drop_duplicates("label") prevents the same game appearing multiple
            # times (once per event row).
            match_opts = (
                team_events_all.dropna(subset=["match_date", "home_team", "away_team"])
                .assign(label=lambda df: df.apply(match_label, axis=1))
                .drop_duplicates("label")
                .sort_values("match_date", ascending=False)["label"]
                .tolist()
            )
            sel_match = st.selectbox("Match:", ["All Matches"] + match_opts)
        else:
            sel_match = "All Matches"

    # ── Row 2: Event types + Period ───────────────────────────────────────────
    # Event types multiselect — can combine e.g. Pass + BallTouch to see
    # total touches.  Column widths [3,1] give more space to the multiselect.
    r2a, r2b = st.columns([3, 1])
    with r2a:
        # Pre-select a sensible default if those types exist in the data.
        default_types = [t for t in ["Pass", "BallTouch", "TackleX"] if t in event_types]
        sel_types = st.multiselect("Event Types:", event_types,
                                   default=default_types or event_types[:3])
    with r2b:
        sel_period = st.selectbox("Period:", ["Full Match", "1st Half", "2nd Half"])

    # ── Apply filters ─────────────────────────────────────────────────────────
    ev = team_events_all.copy()

    # Match filter: compare the computed label string against the selection.
    if sel_match != "All Matches":
        ev_labels = ev.apply(match_label, axis=1)
        ev = ev[ev_labels == sel_match]

    # Event type filter: keep only rows whose 'type' is in the selected list.
    if sel_types:
        ev = ev[ev["type"].isin(sel_types)]

    # Period filter: WhoScored stores "FirstHalf" and "SecondHalf" as strings —
    # map the human-readable dropdown value to the correct string in the data.
    if sel_period != "Full Match" and "period" in ev.columns:
        target = "FirstHalf" if sel_period == "1st Half" else "SecondHalf"
        ev = ev[ev["period"] == target]

    # Drop rows with missing coordinates — can't plot them.
    valid = ev.dropna(subset=["x", "y"])
    st.metric("Events plotted", len(valid))

    if valid.empty:
        st.warning("No events match the selected filters.")
        st.stop()

    # ── Pitch and KDE heatmap ─────────────────────────────────────────────────
    pitch = Pitch(
        pitch_type="opta",
        pitch_color=PITCH_GREEN, line_color="white",
        line_zorder=2, linewidth=1.5,
    )
    fig, ax = pitch.draw(figsize=(12, 7))

    if len(valid) >= 5:
        # kdeplot fits a 2D Kernel Density Estimate over the (x, y) coordinates.
        # cmap="hot" goes black → red → yellow, making dense areas bright yellow.
        # thresh=0.02 hides the bottom 2% of density (sparse outer regions).
        # levels=15 controls the smoothness of the density bands.
        pitch.kdeplot(
            valid["x"].values,
            (100 - valid["y"]).values,   # flip y to match mplsoccer opta orientation
            ax=ax,
            cmap="hot", fill=True,
            alpha=0.72, thresh=0.02, levels=15,
            zorder=2,
        )
    else:
        # KDE needs at least a handful of spread-out points to work.
        # For very sparse filters, fall back to individual event dots instead.
        pitch.scatter(
            valid["x"].values,
            (100 - valid["y"]).values,
            ax=ax, s=120, color="tomato",
            edgecolors="white", linewidths=0.6,
            zorder=3,
        )
        ax.text(50, 95, f"Only {len(valid)} events — showing scatter",
               ha="center", fontsize=10, color="white",
               bbox=dict(boxstyle="round", facecolor="#333", alpha=0.8, pad=0.4))

    # Build a descriptive title from all active filter selections.
    type_str  = ", ".join(sel_types) if sel_types else "All Events"
    match_str = sel_match if sel_match != "All Matches" else "All Matches"
    ax.set_title(f"{sel_team}  ·  {type_str}  ·  {match_str}  ({sel_period})",
                fontsize=12, fontweight="bold", color="white", pad=12)
    fig.patch.set_facecolor("#0e1117")
    st.pyplot(fig)

    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight", dpi=150)
    st.download_button("📥 Heatmap (PNG)", img.getvalue(),
                      file_name="event_heatmap.png", mime="image/png")

    # ── Event type breakdown bar chart ────────────────────────────────────────
    # Shows a count of every event type for this team / period so you can see
    # which actions dominate their game.  head(20) limits to the top 20 types.
    # The [::-1] reversal puts the most common type at the top of the bar chart.
    st.markdown("---")
    st.subheader(f"{sel_team} — Event Type Breakdown ({sel_period})")
    type_counts = (
        events[events["team"] == sel_team]
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
        # ── Goals vs xG scatter ───────────────────────────────────────────────
        # Each dot is one player.  The dashed diagonal line represents perfect
        # xG prediction (Goals = xG).  Players above the line are
        # overperforming their xG (scoring more than expected); players below
        # are underperforming.  Dot colour also encodes xG (green=high).
        # The top 6 by xG are labelled to identify key players easily.
        st.subheader("Goals vs xG")
        if {"goals", "xg"}.issubset(filtered.columns) and len(filtered) > 1:
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(
                filtered["xg"], filtered["goals"],
                s=60, alpha=0.75,
                c=filtered["xg"], cmap="RdYlGn",   # red=low xG, green=high
                edgecolors="white", linewidths=0.4,
            )
            max_v = max(filtered["xg"].max(), filtered["goals"].max()) * 1.08
            # y=x reference line — points above it are overperformers
            ax.plot([0, max_v], [0, max_v], "w--", alpha=0.4, label="xG = Goals")
            ax.set_xlabel("xG")
            ax.set_ylabel("Goals")
            ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
            ax.grid(alpha=0.25)
            for _, p in filtered.nlargest(6, "xg").iterrows():
                ax.annotate(p["player"], (p["xg"], p["goals"]),
                           fontsize=7, color="white",
                           xytext=(4, 4), textcoords="offset points")
            plt.colorbar(sc, ax=ax, label="xG")
            dark_fig_style(fig, ax)
            plt.tight_layout()
            st.pyplot(fig)

    with c2:
        # ── Top 15 by xG bar chart ────────────────────────────────────────────
        # nlargest(15, "xg") selects the 15 players with the highest season xG.
        # invert_yaxis() puts the highest xG player at the top of the chart.
        st.subheader("Top 15 by xG")
        if "xg" in filtered.columns:
            top15 = filtered.nlargest(15, "xg")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(top15["player"], top15["xg"], color=PURPLE)
            ax.set_xlabel("xG")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            dark_fig_style(fig, ax)
            plt.tight_layout()
            st.pyplot(fig)

    # ── Assists vs xA scatter ─────────────────────────────────────────────────
    # Same concept as the Goals vs xG chart but for chance creation.
    # xA (expected assists) measures the quality of passes that led to shots.
    # Players above the diagonal are converting their key passes into assists
    # at a higher rate than the model predicts.
    if {"assists", "xa"}.issubset(filtered.columns) and len(filtered) > 1:
        st.markdown("---")
        st.subheader("Assists vs xA")
        fig, ax = plt.subplots(figsize=(10, 4))
        sc = ax.scatter(filtered["xa"], filtered["assists"],
                       s=60, alpha=0.75, c=filtered["xa"], cmap="Blues",
                       edgecolors="white", linewidths=0.4)
        max_v = max(filtered["xa"].max(), filtered["assists"].max()) * 1.08
        ax.plot([0, max_v], [0, max_v], "w--", alpha=0.4, label="xA = Assists")
        ax.set_xlabel("xA")
        ax.set_ylabel("Assists")
        ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
        ax.grid(alpha=0.25)
        for _, p in filtered.nlargest(5, "xa").iterrows():
            ax.annotate(p["player"], (p["xa"], p["assists"]),
                       fontsize=7, color="white",
                       xytext=(4, 4), textcoords="offset points")
        plt.colorbar(sc, ax=ax, label="xA")
        dark_fig_style(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)


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
    st.title("📈 Team xG Analysis")

    ms = S["match_summary"]
    needed = {"home_team", "away_team", "home_xg", "away_xg", "home_goals", "away_goals"}
    if ms.empty or not needed.issubset(ms.columns):
        st.warning("Match summary data is incomplete — re-run `build_processed.py`.")
        st.stop()

    # ── Build team-level aggregates ────────────────────────────────────────────
    # match_summary has one row per match with home_ and away_ prefixed columns.
    # To get per-team totals we need to handle each team's home AND away games.
    # We do this by splitting the table into four views and summing by team.

    # Each team's xG when they were the home side
    home_for  = ms[["home_team", "home_xg",  "home_goals"]].rename(
        columns={"home_team": "team", "home_xg": "xgf", "home_goals": "gf"})
    # Each team's xG when they were the away side
    away_for  = ms[["away_team", "away_xg",  "away_goals"]].rename(
        columns={"away_team": "team", "away_xg": "xgf", "away_goals": "gf"})
    # xG conceded as the home team = the away team's xG
    home_agst = ms[["home_team", "away_xg",  "away_goals"]].rename(
        columns={"home_team": "team", "away_xg": "xga", "away_goals": "ga"})
    # xG conceded as the away team = the home team's xG
    away_agst = ms[["away_team", "home_xg",  "home_goals"]].rename(
        columns={"away_team": "team", "home_xg": "xga", "home_goals": "ga"})

    # Stack home+away rows and group by team to sum across the full season.
    agg = (
        pd.concat([home_for, away_for]).groupby("team")[["xgf", "gf"]].sum()
        .join(pd.concat([home_agst, away_agst]).groupby("team")[["xga", "ga"]].sum())
        .reset_index()
    )
    agg["xgd"]        = agg["xgf"] - agg["xga"]     # net expected goal difference
    agg["g_minus_xg"] = agg["gf"]  - agg["xgf"]     # goals scored above/below model
    agg = agg.sort_values("xgf", ascending=False).reset_index(drop=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Team xG Table")
    st.dataframe(
        agg.rename(columns={"xgf": "xGF", "gf": "GF", "xga": "xGA",
                             "ga": "GA", "xgd": "xGD", "g_minus_xg": "G − xG"})
        .round(2),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        # ── xGF vs xGA scatter ────────────────────────────────────────────────
        # The four quadrants tell different stories:
        #   Top-right:    High xGF, high xGA — exciting but open teams
        #   Top-left:     Low xGF, high xGA  — struggling teams
        #   Bottom-right: High xGF, low xGA  — dominant teams (top of table)
        #   Bottom-left:  Low xGF, low xGA   — cautious / defensive teams
        #
        # Y-axis is inverted so low xGA (good defence) appears at the TOP,
        # meaning the best teams naturally float to the top-right corner.
        # Dashed cross-hairs mark the league average for both axes.
        # Dot colour encodes xGD: green=positive, red=negative.
        st.subheader("xGF vs xGA")
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(
            agg["xgf"], agg["xga"],
            s=90, alpha=0.85,
            c=agg["xgd"], cmap="RdYlGn",
            edgecolors="white", linewidths=0.5,
        )
        # League-average reference lines
        ax.axhline(agg["xga"].mean(), color="white", linestyle="--", alpha=0.35, lw=1)
        ax.axvline(agg["xgf"].mean(), color="white", linestyle="--", alpha=0.35, lw=1)
        ax.invert_yaxis()   # lower xGA = better defence → appears at top
        ax.set_xlabel("xG For")
        ax.set_ylabel("xG Against  (lower = better ↑)")
        ax.grid(alpha=0.25)
        for _, r in agg.iterrows():
            ax.annotate(r["team"], (r["xgf"], r["xga"]),
                       fontsize=7, color="white",
                       xytext=(4, 4), textcoords="offset points")
        plt.colorbar(sc, ax=ax, label="xGD")
        dark_fig_style(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        # ── Overperformance bar chart ─────────────────────────────────────────
        # Shows Goals − xG for every team, sorted low to high.
        # Green bars = scored more goals than the model expected (lucky or clinical).
        # Red bars   = scored fewer goals than expected (unlucky or wasteful).
        # This is one of the most useful charts for predicting regression —
        # teams far into the green are likely to slow down, and vice versa.
        st.subheader("Goals vs xG (Overperformance)")
        agg_s = agg.sort_values("g_minus_xg")
        colors = [GREEN if v >= 0 else "#ff4444" for v in agg_s["g_minus_xg"]]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(agg_s["team"], agg_s["g_minus_xg"], color=colors, alpha=0.88)
        ax.axvline(0, color="white", linewidth=1)   # zero line = perfectly on xG
        ax.set_xlabel("Goals − xG")
        ax.grid(axis="x", alpha=0.3)
        dark_fig_style(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Grouped bar: xGF vs actual goals per team ─────────────────────────────
    # Side-by-side bars let you quickly scan which teams over- or under-scored
    # relative to their expected goals across the whole season.
    # np.arange() creates evenly spaced x positions for the team labels.
    # w=0.35 is the bar width; bars are offset by ±w/2 to sit next to each other.
    st.markdown("---")
    st.subheader("Expected vs Actual Goals — all teams")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(agg))
    w = 0.35
    ax.bar(x - w / 2, agg["xgf"], w, label="xGF",  color=PURPLE, alpha=0.85)
    ax.bar(x + w / 2, agg["gf"],  w, label="Goals", color=GREEN,  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["team"], rotation=45, ha="right", color="white")
    ax.set_ylabel("Goals / xG")
    ax.legend(facecolor="#1a1a1a", labelcolor="white")
    ax.grid(axis="y", alpha=0.3)
    dark_fig_style(fig, ax)
    plt.tight_layout()
    st.pyplot(fig)


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
    st.subheader(f"{sel_gt} — Season Grades ({sel_gs})")

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

        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot the four sub-grade lines
        grade_lines = [
            ("roll_overall_grade",  "⭐ Overall",  "#ffffff", 2.5),
            ("roll_attack_grade",   "⚔️ Attack",   "#00ff85", 1.5),
            ("roll_defense_grade",  "🛡️ Defense",  "#4da6ff", 1.5),
            ("roll_style_grade",    "🎨 Style",    "#ffd700", 1.2),
        ]
        for col, lbl, color, lw in grade_lines:
            if col in team_rolling.columns:
                ax.plot(team_rolling["gw"], team_rolling[col],
                        label=lbl, color=color, linewidth=lw, marker="o",
                        markersize=3, alpha=0.9)

        # Shade background green (good form) / red (poor form) zones
        ax.axhspan(7, 10, alpha=0.06, color="#00c853")
        ax.axhspan(1,  4, alpha=0.06, color="#ff1744")
        ax.axhline(5.5, color="white", linewidth=0.5, linestyle="--", alpha=0.3)

        ax.set_xlim(1, len(team_rolling))
        ax.set_ylim(1, 10)
        ax.set_xlabel("Gameweek", color="white")
        ax.set_ylabel("Grade (1–10)", color="white")
        ax.set_title(
            f"Rolling 5-match form — {sel_gt}",
            color="white", fontsize=12
        )
        ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9,
                  loc="upper left")
        ax.grid(alpha=0.2)

        # Annotate W/D/L results above the x-axis
        result_colors = {"W": "#00c853", "D": "#ffd700", "L": "#ff1744"}
        for _, row in team_rolling.iterrows():
            res = row.get("result", "")
            if res in result_colors:
                ax.text(row["gw"], 1.15, res, ha="center", va="bottom",
                        fontsize=7, color=result_colors[res], fontweight="bold")

        dark_fig_style(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)
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

    with st.spinner("Computing player grades…"):
        pg = _get_player_grades(ps)

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
    disp_cols = [c for c in
                 ["player", "team", "position", "games", "time",
                  "goals", "assists",
                  "npxg_p90", "xa_p90", "kp_p90",
                  "attack_grade", "creativity_grade", "overall_grade"]
                 if c in pgf.columns]

    tbl = pgf[disp_cols].sort_values("overall_grade", ascending=False).copy()

    # Round per-90 metrics and grade columns
    for c in ["npxg_p90", "xa_p90", "kp_p90",
              "attack_grade", "creativity_grade", "overall_grade"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].round(2)

    st.dataframe(
        tbl.rename(columns={
            "player": "Player", "team": "Team", "position": "Pos",
            "games": "Apps", "time": "Mins",
            "goals": "Goals", "assists": "Assists",
            "npxg_p90": "npxG/90", "xa_p90": "xA/90", "kp_p90": "KP/90",
            "attack_grade": "Attack", "creativity_grade": "Creativity",
            "overall_grade": "Overall",
        }),
        use_container_width=True, hide_index=True, height=450,
    )

    st.markdown("---")

    # ── Top 15 by overall grade (bar chart) ───────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Top 15 by Overall Grade")
        top15 = pgf.nlargest(15, "overall_grade")
        if not top15.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            colors_bar = [
                "#00c853" if g >= 7 else ("#ffd600" if g >= 4.5 else "#ff1744")
                for g in top15["overall_grade"]
            ]
            ax.barh(top15["player"], top15["overall_grade"], color=colors_bar, alpha=0.9)
            ax.set_xlabel("Overall Grade (1–10)")
            ax.set_xlim(1, 10)
            ax.invert_yaxis()
            ax.axvline(5.5, color="white", linestyle="--", alpha=0.3, lw=1)
            ax.grid(axis="x", alpha=0.25)
            dark_fig_style(fig, ax)
            plt.tight_layout()
            st.pyplot(fig)

    with c2:
        # ── Attack vs Creativity scatter ──────────────────────────────────────
        # Each dot is a player.  Top players are labelled.
        st.subheader("Attack vs Creativity")
        if {"attack_grade", "creativity_grade"}.issubset(pgf.columns) and len(pgf) > 2:
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(
                pgf["attack_grade"], pgf["creativity_grade"],
                c=pgf["overall_grade"], cmap="RdYlGn",
                s=60, alpha=0.75, edgecolors="white", linewidths=0.4,
                vmin=1, vmax=10,
            )
            # League average reference lines
            ax.axhline(pgf["creativity_grade"].mean(), color="white",
                       linestyle="--", alpha=0.3, lw=1)
            ax.axvline(pgf["attack_grade"].mean(), color="white",
                       linestyle="--", alpha=0.3, lw=1)
            ax.set_xlabel("Attack Grade")
            ax.set_ylabel("Creativity Grade")
            ax.set_xlim(1, 10)
            ax.set_ylim(1, 10)
            # Label top 8 by overall grade
            for _, p in pgf.nlargest(8, "overall_grade").iterrows():
                ax.annotate(p["player"].split()[-1],   # surname only
                            (p["attack_grade"], p["creativity_grade"]),
                            fontsize=7, color="white",
                            xytext=(4, 4), textcoords="offset points")
            plt.colorbar(sc, ax=ax, label="Overall Grade")
            ax.grid(alpha=0.2)
            dark_fig_style(fig, ax)
            plt.tight_layout()
            st.pyplot(fig)


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

    st.markdown(f"### {home_team}  vs  {away_team}")
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
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources**")
st.sidebar.markdown("- Understat · ESPN · WhoScored")
st.sidebar.markdown("---")
st.sidebar.info("💡 Run `python build_processed.py` after each collection cycle to refresh.")
