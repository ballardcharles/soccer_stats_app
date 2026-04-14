"""
dashboard.py
------------
Premier League Analytics Dashboard
Data: Understat · ESPN · WhoScored  →  data/processed/

Run with:
    streamlit run dashboard.py
"""

import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from mplsoccer import Pitch, VerticalPitch

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Premier League Analytics",
    layout="wide",
    page_icon="⚽",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #e8e8e8; }
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Colour constants
PURPLE = "#37003c"   # Premier League purple
GREEN  = "#00ff85"   # Premier League green
PITCH_GREEN = "#1a472a"

PROCESSED = "data/processed"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    def safe_read(name: str) -> pd.DataFrame:
        try:
            return pd.read_csv(f"{PROCESSED}/{name}", low_memory=False)
        except FileNotFoundError:
            return pd.DataFrame()

    shots         = safe_read("shots.csv")
    events        = safe_read("events.csv")
    match_summary = safe_read("match_summary.csv")
    player_season = safe_read("player_season.csv")
    lineups       = safe_read("lineups.csv")

    # Parse dates once at load time
    for df in (shots, events, match_summary, lineups):
        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    return dict(
        shots=shots,
        events=events,
        match_summary=match_summary,
        player_season=player_season,
        lineups=lineups,
    )


with st.spinner("Loading data…"):
    D = load_data()

if all(v.empty for v in D.values()):
    st.error("No processed data found in `data/processed/`. "
             "Run `python build_processed.py` first.")
    st.stop()


# ── Shared helpers ────────────────────────────────────────────────────────────

def team_list(df: pd.DataFrame, *cols: str) -> list[str]:
    names: set = set()
    for c in cols:
        if c in df.columns:
            names.update(df[c].dropna().unique())
    return sorted(names)


def match_label(row) -> str:
    try:
        d = pd.to_datetime(row["match_date"]).strftime("%d %b")
    except Exception:
        d = "?"
    return f"{d}  {row['home_team']} v {row['away_team']}"


def dark_fig_style(fig, *axes):
    """Apply consistent dark background to a matplotlib figure."""
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#121212")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        if ax.get_title():
            ax.set_title(ax.get_title(), color="white")


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("⚽ PL Analytics")
st.sidebar.markdown("---")

view = st.sidebar.selectbox("View:", [
    "🎯 Shot Maps",
    "🔥 Event Heatmaps",
    "📊 Match Analysis",
    "👤 Player Stats",
    "📈 xG Analysis",
])

st.sidebar.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# 🎯  SHOT MAPS
# ══════════════════════════════════════════════════════════════════════════════

if view == "🎯 Shot Maps":
    st.title("🎯 Shot Maps")

    shots = D["shots"]
    if shots.empty:
        st.warning("No shot data — run the collection pipeline.")
        st.stop()

    # ── Filters ──────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)

    with f1:
        sel_team = st.selectbox("Team:", team_list(shots, "team"))
        team_shots = shots[shots["team"] == sel_team].copy()

    with f2:
        players = ["All Players"] + sorted(team_shots["player"].dropna().unique())
        sel_player = st.selectbox("Player:", players)

    with f3:
        if "match_date" in team_shots.columns and "opponent" in team_shots.columns:
            # shots.csv has team/opponent, not home_team/away_team — build label from those
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

    # Apply filters
    filtered = team_shots.copy()
    if sel_player != "All Players":
        filtered = filtered[filtered["player"] == sel_player]
    if sel_match != "All Matches":
        shot_labels = (
            filtered["match_date"].dt.strftime("%d %b") + "  vs " + filtered["opponent"]
        )
        filtered = filtered[shot_labels == sel_match]

    if "result" in filtered.columns:
        all_results = sorted(filtered["result"].dropna().unique())
        sel_results = st.multiselect("Shot Results:", all_results, default=all_results)
        filtered = filtered[filtered["result"].isin(sel_results)]

    viz_type = st.radio("Visualization:", ["Shot Map", "Heat Map"], horizontal=True)

    # ── Metrics ──────────────────────────────────────────────────────────────
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

    # ── Pitch viz ─────────────────────────────────────────────────────────────
    # Coordinates are in WhoScored/opta [0,100]:
    #   x = 0 own goal → 100 attacking goal
    #   y = 0 top touchline → 100 bottom touchline
    # VerticalPitch(half=True) shows x 50→100 (attacking half).
    # mplsoccer opta y is bottom→top so we flip: y_plot = 100 - y
    pitch = VerticalPitch(
        pitch_type="opta", half=True,
        pitch_color=PITCH_GREEN, line_color="white",
        line_zorder=2, linewidth=1.5,
    )
    fig, ax = pitch.draw(figsize=(8, 7))

    if viz_type == "Shot Map":
        RESULT_STYLE = {
            "Goal":        dict(color="lime",    marker="*",  base_s=300),
            "SavedShot":   dict(color="yellow",  marker="o",  base_s=80),
            "BlockedShot": dict(color="orange",  marker="s",  base_s=80),
            "MissedShots": dict(color="#ff4444", marker="x",  base_s=80),
        }
        for _, shot in filtered.iterrows():
            sx  = shot.get("x", 75)
            sy  = 100 - shot.get("y", 50)     # flip y for correct orientation
            res = shot.get("result", "MissedShots")
            xg  = float(shot.get("xg", 0.05))
            st_ = RESULT_STYLE.get(res, dict(color="grey", marker="x", base_s=60))
            pitch.scatter(
                sx, sy,
                s=xg * 600 + st_["base_s"],
                c=st_["color"],
                marker=st_["marker"],
                alpha=0.85,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
                ax=ax,
            )

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
        if len(filtered) >= 5:
            pitch.kdeplot(
                filtered["x"].values,
                (100 - filtered["y"]).values,
                ax=ax, cmap="Reds", fill=True,
                alpha=0.75, thresh=0.05, levels=10,
            )
        ax.text(50, 52, f"Total Shots: {n_shots}", fontsize=11, ha="center",
               fontweight="bold",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, pad=0.4))

    # Title
    title = sel_team
    if sel_player != "All Players":  title += f" — {sel_player}"
    if sel_match  != "All Matches":  title += f"  ·  {sel_match}"
    ax.set_title(title, fontsize=12, fontweight="bold", color="white", pad=10)
    fig.patch.set_facecolor("#0e1117")
    st.pyplot(fig)

    # Download buttons
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

    # ── Supporting charts ─────────────────────────────────────────────────────
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

    st.subheader("Shot Details")
    disp = [c for c in ["player", "minute", "result", "xg", "shot_type", "situation"]
            if c in filtered.columns]
    st.dataframe(filtered[disp].sort_values("xg", ascending=False),
                use_container_width=True, height=350, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🔥  EVENT HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

elif view == "🔥 Event Heatmaps":
    st.title("🔥 Event Heatmaps")

    events = D["events"]
    if events.empty:
        st.warning("No event data — run the WhoScored collector.")
        st.stop()

    event_types = sorted(events["type"].dropna().unique()) if "type" in events.columns else []

    f1, f2, f3 = st.columns(3)
    with f1:
        sel_team = st.selectbox("Team:", team_list(events, "team"))
    with f2:
        default_types = [t for t in ["Pass", "BallTouch", "TackleX"] if t in event_types]
        sel_types = st.multiselect("Event Types:", event_types,
                                   default=default_types or event_types[:3])
    with f3:
        sel_period = st.selectbox("Period:", ["Full Match", "1st Half", "2nd Half"])

    # Filter
    ev = events[events["team"] == sel_team].copy()
    if sel_types:
        ev = ev[ev["type"].isin(sel_types)]
    if sel_period == "1st Half" and "period" in ev.columns:
        ev = ev[ev["period"] == 1]
    elif sel_period == "2nd Half" and "period" in ev.columns:
        ev = ev[ev["period"] == 2]

    valid = ev.dropna(subset=["x", "y"])
    st.metric("Events plotted", len(valid))

    if len(valid) < 10:
        st.warning(f"Not enough events ({len(valid)}) to generate a heatmap. "
                   "Try broadening the filters.")
        st.stop()

    pitch = Pitch(
        pitch_type="opta",
        pitch_color=PITCH_GREEN, line_color="white",
        line_zorder=2, linewidth=1.5,
    )
    fig, ax = pitch.draw(figsize=(12, 7))

    pitch.kdeplot(
        valid["x"].values,
        (100 - valid["y"]).values,   # flip y for correct orientation
        ax=ax,
        cmap="hot", fill=True,
        alpha=0.72, thresh=0.02, levels=15,
        zorder=2,
    )

    type_str = ", ".join(sel_types) if sel_types else "All Events"
    ax.set_title(f"{sel_team}  ·  {type_str}  ({sel_period})",
                fontsize=13, fontweight="bold", color="white", pad=12)
    fig.patch.set_facecolor("#0e1117")
    st.pyplot(fig)

    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight", dpi=150)
    st.download_button("📥 Heatmap (PNG)", img.getvalue(),
                      file_name="event_heatmap.png", mime="image/png")

    # Event type breakdown for this team
    st.markdown("---")
    st.subheader(f"{sel_team} — Event Type Breakdown ({sel_period})")
    type_counts = (
        events[events["team"] == sel_team]
        .pipe(lambda df: df[df["period"] == 1] if sel_period == "1st Half" and "period" in df.columns
              else (df[df["period"] == 2] if sel_period == "2nd Half" and "period" in df.columns else df))
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

elif view == "📊 Match Analysis":
    st.title("📊 Match Analysis")

    ms = D["match_summary"]
    if ms.empty:
        st.warning("No match data available.")
        st.stop()

    ms_sorted = ms.sort_values("match_date", ascending=False).copy()
    ms_sorted["_label"] = ms_sorted.apply(match_label, axis=1)

    sel_label = st.selectbox("Select Match:", ms_sorted["_label"].tolist())
    row = ms_sorted[ms_sorted["_label"] == sel_label].iloc[0]

    st.markdown("---")

    # ── Match card ────────────────────────────────────────────────────────────
    home_col, mid_col, away_col = st.columns([4, 2, 4])

    with home_col:
        st.markdown(f"### 🏠 {row['home_team']}")
        st.metric("Goals", int(row.get("home_goals", 0)))
        if "home_xg" in row:
            st.metric("xG", f"{row['home_xg']:.2f}")
        if "forecast_win" in row:
            st.metric("Pre-match Win %", f"{row['forecast_win']*100:.0f}%")

    with mid_col:
        date_str = pd.to_datetime(row["match_date"]).strftime("%d %b %Y")
        score = f"{int(row.get('home_goals',0))} – {int(row.get('away_goals',0))}"
        st.markdown(
            f"<div style='text-align:center; padding-top:24px'>"
            f"<p style='color:#aaa; font-size:13px; margin-bottom:4px'>{date_str}</p>"
            f"<h2 style='color:{GREEN}; margin:0'>{score}</h2>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with away_col:
        st.markdown(f"### ✈️ {row['away_team']}")
        st.metric("Goals", int(row.get("away_goals", 0)))
        if "away_xg" in row:
            st.metric("xG", f"{row['away_xg']:.2f}")
        if "forecast_loss" in row:
            st.metric("Pre-match Win %", f"{row['forecast_loss']*100:.0f}%")

    st.markdown("---")

    # ── xG comparison ─────────────────────────────────────────────────────────
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
        for bar, val in zip(bars, [a_xg, h_xg]):
            ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
                   f"{val:.2f}", va="center", fontsize=12, fontweight="bold", color="white")
        ax.set_xlabel("Expected Goals (xG)")
        ax.grid(axis="x", alpha=0.3)
        dark_fig_style(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Pre-match forecast ────────────────────────────────────────────────────
    if all(k in row.index for k in ["forecast_win", "forecast_draw", "forecast_loss"]):
        st.subheader("Pre-Match Forecast")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric(f"{row['home_team']} Win", f"{row['forecast_win']*100:.0f}%")
        fc2.metric("Draw",                    f"{row['forecast_draw']*100:.0f}%")
        fc3.metric(f"{row['away_team']} Win", f"{row['forecast_loss']*100:.0f}%")

    # ── Per-match shot map ─────────────────────────────────────────────────────
    shots = D["shots"]
    if not shots.empty and "match_date" in shots.columns:
        match_date_val = pd.to_datetime(row["match_date"]).date()
        match_shots = shots[
            (shots["match_date"].dt.date == match_date_val) &
            (shots["team"].isin([row["home_team"], row["away_team"]]))
        ]
        if not match_shots.empty:
            st.markdown("---")
            st.subheader("Shot Map")
            pitch = VerticalPitch(
                pitch_type="opta", half=True,
                pitch_color=PITCH_GREEN, line_color="white",
            )
            fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(14, 7))
            for ax, team in zip(axes, [row["home_team"], row["away_team"]]):
                t_shots = match_shots[match_shots["team"] == team]
                for _, shot in t_shots.iterrows():
                    res   = shot.get("result", "")
                    color = "lime" if res == "Goal" else ("yellow" if res == "SavedShot" else "#ff4444")
                    pitch.scatter(
                        shot["x"], 100 - shot["y"],
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

elif view == "👤 Player Stats":
    st.title("👤 Player Season Stats")

    ps = D["player_season"]
    if ps.empty:
        st.warning("No player data — run the Understat collector.")
        st.stop()

    f1, f2, f3 = st.columns(3)
    with f1:
        teams = ["All Teams"] + sorted(ps["primary_team"].dropna().unique())
        sel_team = st.selectbox("Team:", teams)
    with f2:
        positions = (["All Positions"] + sorted(ps["position"].dropna().unique())
                     if "position" in ps.columns else ["All Positions"])
        sel_pos = st.selectbox("Position:", positions)
    with f3:
        min_games = st.slider("Min. Appearances:", 0, 30, 5)

    filtered = ps.copy()
    if sel_team != "All Teams":
        filtered = filtered[filtered["primary_team"] == sel_team]
    if sel_pos != "All Positions" and "position" in filtered.columns:
        filtered = filtered[filtered["position"] == sel_pos]
    if "games" in filtered.columns:
        filtered = filtered[filtered["games"] >= min_games]

    st.metric("Players shown", len(filtered))

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
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(
                filtered["xg"], filtered["goals"],
                s=60, alpha=0.75,
                c=filtered["xg"], cmap="RdYlGn",
                edgecolors="white", linewidths=0.4,
            )
            max_v = max(filtered["xg"].max(), filtered["goals"].max()) * 1.08
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

    # xA scatter
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

elif view == "📈 xG Analysis":
    st.title("📈 Team xG Analysis")

    ms = D["match_summary"]
    needed = {"home_team", "away_team", "home_xg", "away_xg", "home_goals", "away_goals"}
    if ms.empty or not needed.issubset(ms.columns):
        st.warning("Match summary data is incomplete — re-run `build_processed.py`.")
        st.stop()

    # Aggregate per-team from both home and away rows
    home_for  = ms[["home_team", "home_xg",   "home_goals"]].rename(columns={"home_team": "team", "home_xg": "xgf",   "home_goals": "gf"})
    away_for  = ms[["away_team", "away_xg",   "away_goals"]].rename(columns={"away_team": "team", "away_xg": "xgf",   "away_goals": "gf"})
    home_agst = ms[["home_team", "away_xg",   "away_goals"]].rename(columns={"home_team": "team", "away_xg": "xga",   "away_goals": "ga"})
    away_agst = ms[["away_team", "home_xg",   "home_goals"]].rename(columns={"away_team": "team", "home_xg": "xga",   "home_goals": "ga"})

    agg = (
        pd.concat([home_for, away_for]).groupby("team")[["xgf", "gf"]].sum()
        .join(pd.concat([home_agst, away_agst]).groupby("team")[["xga", "ga"]].sum())
        .reset_index()
    )
    agg["xgd"]     = agg["xgf"] - agg["xga"]
    agg["g_minus_xg"] = agg["gf"] - agg["xgf"]
    agg = agg.sort_values("xgf", ascending=False).reset_index(drop=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    st.subheader("Team xG Table")
    st.dataframe(
        agg.rename(columns={"xgf": "xGF", "gf": "GF", "xga": "xGA",
                             "ga": "GA", "xgd": "xGD", "g_minus_xg": "G − xG"})
        .round(2),
        use_container_width=True, hide_index=True,
    )

    # ── Scatter: xGF vs xGA ───────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("xGF vs xGA")
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(
            agg["xgf"], agg["xga"],
            s=90, alpha=0.85,
            c=agg["xgd"], cmap="RdYlGn",
            edgecolors="white", linewidths=0.5,
        )
        ax.axhline(agg["xga"].mean(), color="white", linestyle="--", alpha=0.35, lw=1)
        ax.axvline(agg["xgf"].mean(), color="white", linestyle="--", alpha=0.35, lw=1)
        ax.invert_yaxis()   # lower xGA = better → top of chart
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
        st.subheader("Goals vs xG (Overperformance)")
        agg_s = agg.sort_values("g_minus_xg")
        colors = [GREEN if v >= 0 else "#ff4444" for v in agg_s["g_minus_xg"]]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(agg_s["team"], agg_s["g_minus_xg"], color=colors, alpha=0.88)
        ax.axvline(0, color="white", linewidth=1)
        ax.set_xlabel("Goals − xG")
        ax.grid(axis="x", alpha=0.3)
        dark_fig_style(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Grouped bar: xGF vs GF per team ──────────────────────────────────────
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


# ── Footer ────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources**")
st.sidebar.markdown("- Understat · ESPN · WhoScored")
st.sidebar.markdown("---")
st.sidebar.info("💡 Run `python build_processed.py` after each collection cycle to refresh.")
