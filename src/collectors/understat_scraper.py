"""
understat_scraper.py
--------------------
Scrapes match and shot data from understat.com using Selenium.
Supports incremental / resume-safe weekly runs — only fetches
shot data for matches not already collected.

How incremental mode works:
    1. Always re-fetches the match list (fast, one page load).
       This is how we discover newly completed matches each week.
    2. Loads the existing shots CSV from disk (if present).
    3. Compares match IDs — only scrapes shots for matches
       that are completed but missing from the shots file.
    4. Appends new shots to the existing file rather than overwriting.

Scheduling (macOS cron example — runs every Monday at 7am):
    crontab -e
    0 7 * * 1 cd /path/to/Soccer_Stats_App && /usr/bin/python3 run_collection.py >> logs/understat.log 2>&1

Requirements:
    pip install selenium
    Chrome must be installed. ChromeDriver managed automatically by Selenium 4.6+.

Usage:
    python run_collection.py
    python -m src.collectors.understat_scraper --shots
    python -m src.collectors.understat_scraper --shots --visible
"""

import json
import re
import time
import random
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

from src.utils import ensure_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://understat.com"

LEAGUE_MAP = {
    "EPL":        "EPL",
    "La_liga":    "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie_A":    "Serie_A",
    "Ligue_1":    "Ligue_1",
    "RFPL":       "RFPL",
}

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------

def _make_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1440,900")
    options.add_argument("--lang=en-GB")
    options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver


def _human_delay(min_s: float = 2.0, max_s: float = 6.0) -> None:
    delay = random.uniform(min_s, max_s)
    print(f"  [waiting {delay:.1f}s]")
    time.sleep(delay)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _wait_for_data(driver: webdriver.Chrome, var_name: str, timeout: int = 20) -> None:
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script(
            f"try {{ return typeof {var_name} !== 'undefined' && {var_name} !== null; }}"
            "catch(e) { return false; }"
        )
    )


def _extract_json_var(driver: webdriver.Chrome, var_name: str) -> list:
    # Primary: pull directly from JS context
    try:
        result = driver.execute_script(f"return JSON.stringify({var_name});")
        if result:
            return json.loads(result)
    except Exception:
        pass

    # Fallback: regex on page source
    html = driver.page_source
    pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
    match = re.search(pattern, html)
    if match:
        raw = match.group(1).encode("utf-8").decode("unicode_escape")
        return json.loads(raw)

    raise ValueError(f"Could not find '{var_name}' on the page — may not have finished loading.")


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_collected_ids(shots_path: Path) -> set:
    """
    Returns the set of match_ids already present in the shots CSV.
    If the file doesn't exist yet, returns an empty set.
    """
    if not shots_path.exists():
        return set()
    try:
        existing = pd.read_csv(shots_path, usecols=["match_id"])
        ids = set(existing["match_id"].astype(str).unique())
        print(f"  Found existing shots file with {len(ids)} matches already collected.")
        return ids
    except Exception as exc:
        print(f"  ⚠ Could not read existing shots file: {exc}. Starting fresh.")
        return set()


def _append_shots(shots_df: pd.DataFrame, shots_path: Path) -> None:
    """
    Appends new shot rows to the existing CSV, or creates it if it doesn't exist.
    Deduplicates by match_id before writing to guard against double-writes on retry.
    """
    if shots_path.exists() and not shots_df.empty:
        try:
            existing_ids = set(
                pd.read_csv(shots_path, usecols=["match_id"])["match_id"].astype(str)
            )
            shots_df = shots_df[~shots_df["match_id"].astype(str).isin(existing_ids)]
            if shots_df.empty:
                print("    (match already in shots file — skipping duplicate write)")
                return
        except Exception as exc:
            print(f"    ⚠ Could not check for duplicates: {exc}. Writing anyway.")
    write_header = not shots_path.exists()
    shots_df.to_csv(shots_path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# League match list
# ---------------------------------------------------------------------------

def _parse_matches(matches_raw: list) -> pd.DataFrame:
    """Parses the raw datesData list into a clean DataFrame."""
    records = []
    for m in matches_raw:
        records.append({
            "match_id":      m.get("id"),
            "date":          m.get("datetime"),
            "home_team":     m.get("h", {}).get("title"),
            "away_team":     m.get("a", {}).get("title"),
            "home_team_id":  m.get("h", {}).get("id"),
            "away_team_id":  m.get("a", {}).get("id"),
            "home_goals":    m.get("goals", {}).get("h"),
            "away_goals":    m.get("goals", {}).get("a"),
            "home_xg":       m.get("xG", {}).get("h"),
            "away_xg":       m.get("xG", {}).get("a"),
            "forecast_win":  m.get("forecast", {}).get("w"),
            "forecast_draw": m.get("forecast", {}).get("d"),
            "forecast_loss": m.get("forecast", {}).get("l"),
            "is_result":     m.get("isResult"),
        })
    df = pd.DataFrame(records)
    if not df.empty and "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def _parse_players(players_raw) -> pd.DataFrame:
    """
    Parses the raw playersData into a clean DataFrame.

    Understat previously returned playersData as a dict keyed by player_id.
    It now returns a list of player dicts where player_id lives inside each
    object. This function handles both shapes for backwards compatibility.
    """
    # Normalise to an iterable of (player_id, player_dict) pairs
    if isinstance(players_raw, dict):
        items = players_raw.items()
    else:
        # List format: player_id is a field inside each object
        items = ((p.get("id", p.get("player_id")), p) for p in players_raw)

    records = []
    for player_id, p in items:
        records.append({
            "player_id":   player_id,
            "player":      p.get("player_name"),
            "team":        p.get("team_title"),
            "position":    p.get("position"),
            "games":       p.get("games"),
            "time":        p.get("time"),
            "goals":       p.get("goals"),
            "assists":     p.get("assists"),
            "shots":       p.get("shots"),
            "key_passes":  p.get("key_passes"),
            "xg":          p.get("xG"),
            "xa":          p.get("xA"),
            "npg":         p.get("npg"),
            "npxg":        p.get("npxG"),
            "npxg_xa":     p.get("npxGxA"),   # absent in newer API; kept for back-compat
            "xg_chain":    p.get("xGChain"),
            "xg_buildup":  p.get("xGBuildup"),
            "yellow_cards": p.get("yellow_cards"),
            "red_cards":   p.get("red_cards"),
        })
    return pd.DataFrame(records)


def scrape_league_page(
    league: str = "EPL",
    season: int = 2024,
    headless: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the league/season page once and extracts BOTH:
        - Match list  (datesData)
        - Player season stats  (playersData)

    Both variables live on the same page so we get them in a single
    browser session — no extra requests needed.

    Returns
    -------
    (matches_df, players_df)
    """
    url = f"{BASE_URL}/league/{league}/{season}"
    print(f"\n--- Fetching league page: {url} ---")

    driver = _make_driver(headless=headless)
    try:
        print("  Visiting homepage...")
        driver.get(BASE_URL)
        _human_delay(2, 4)

        print("  Loading season page...")
        driver.get(url)

        print("  Waiting for JavaScript data...")
        _wait_for_data(driver, "datesData", timeout=20)
        _human_delay(1, 2)

        print("  Extracting match data...")
        matches_raw = _extract_json_var(driver, "datesData")

        print("  Extracting player season data...")
        try:
            players_raw = _extract_json_var(driver, "playersData")
        except ValueError:
            print("  ⚠ playersData not found — skipping player stats.")
            players_raw = {}

    finally:
        driver.quit()

    matches_df = _parse_matches(matches_raw)
    players_df = _parse_players(players_raw) if players_raw else pd.DataFrame()

    completed = int(matches_df["is_result"].sum()) if "is_result" in matches_df.columns else "?"
    print(f"  Matches: {len(matches_df)} ({completed} completed)")
    print(f"  Players: {len(players_df)}")

    return matches_df, players_df


def scrape_league_matches(
    league: str = "EPL",
    season: int = 2024,
    headless: bool = True,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper — returns only the match DataFrame.
    Use scrape_league_page() directly if you also want player stats.
    """
    matches_df, _ = scrape_league_page(league=league, season=season, headless=headless)
    return matches_df


# ---------------------------------------------------------------------------
# Per-match shot data
# ---------------------------------------------------------------------------

def scrape_match_shots(match_id: str, driver: webdriver.Chrome) -> pd.DataFrame:
    """Scrapes shot-level data for a single match using an existing driver."""
    driver.get(f"{BASE_URL}/match/{match_id}")
    _wait_for_data(driver, "shotsData", timeout=20)
    _human_delay(1, 2)

    shots_raw = _extract_json_var(driver, "shotsData")
    records = []
    for side in ("h", "a"):
        for shot in shots_raw.get(side, []):
            records.append({
                "match_id":  match_id,
                "side":      "home" if side == "h" else "away",
                "player":    shot.get("player"),
                "player_id": shot.get("player_id"),
                "minute":    shot.get("minute"),
                "xg":        shot.get("xG"),
                "result":    shot.get("result"),
                "shot_type": shot.get("shotType"),
                "situation": shot.get("situation"),
                "x":         shot.get("X"),
                "y":         shot.get("Y"),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def pull_understat_data(
    league: str = "EPL",
    season: int = 2024,
    include_shots: bool = False,
    max_matches: int = None,
    headless: bool = True,
) -> dict:
    """
    Incremental scrape: always refreshes the match list and player
    season stats, then only fetches shot data for new matches.

    Parameters
    ----------
    league        : Understat league code (default 'EPL').
    season        : Season start year (default 2024 → 2024/25).
    include_shots : Also scrape shot data for new completed matches.
    max_matches   : Cap new shot scrapes per run (useful for testing).
    headless      : Run Chrome headlessly. Set False to watch the browser.
    """
    ensure_dir("data/raw_understat")
    ensure_dir("logs")
    results = {}

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*50}")
    print(f"Understat scrape started: {run_time}")
    print(f"League: {league}  |  Season: {season}/{season+1}")
    print(f"{'='*50}")

    # --- Always refresh match list + player season stats (one page load) ---
    matches_df, players_df = scrape_league_page(league=league, season=season, headless=headless)

    matches_path = f"data/raw_understat/{league}_{season}_matches.csv"
    matches_df.to_csv(matches_path, index=False)
    print(f"Saved match list → {matches_path}")
    results["matches"] = matches_df

    if not players_df.empty:
        players_path = f"data/raw_understat/{league}_{season}_players.csv"
        players_df.to_csv(players_path, index=False)
        print(f"Saved player stats → {players_path}")
        results["players"] = players_df

    # --- Incremental shot data ---
    if include_shots:
        shots_path = Path(f"data/raw_understat/{league}_{season}_shots.csv")

        # Find out which matches we've already collected shots for
        already_collected = _load_collected_ids(shots_path)

        # Only process completed matches we haven't seen yet
        completed = matches_df[matches_df["is_result"] == True].copy()
        new_matches = completed[
            ~completed["match_id"].astype(str).isin(already_collected)
        ].copy()

        if new_matches.empty:
            print("\n✓ No new matches to collect — already up to date.")
            return results

        if max_matches:
            new_matches = new_matches.head(max_matches)

        print(f"\n--- {len(new_matches)} new matches to scrape shots for ---")
        if already_collected:
            print(f"    (skipping {len(already_collected)} already collected)")

        driver = _make_driver(headless=headless)
        new_shot_count = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3

        try:
            for i, (_, row) in enumerate(new_matches.iterrows(), start=1):
                match_id = str(row["match_id"])
                print(f"  [{i}/{len(new_matches)}] {row['home_team']} vs {row['away_team']} (id={match_id})")
                try:
                    shots_df = scrape_match_shots(match_id, driver)
                    # Append directly to disk after each match —
                    # means progress is saved even if the run is interrupted
                    _append_shots(shots_df, shots_path)
                    new_shot_count += len(shots_df)
                    consecutive_failures = 0
                    print(f"    ✓ {len(shots_df)} shots saved")

                except Exception as exc:
                    consecutive_failures += 1
                    is_session_crash = any(msg in str(exc) for msg in [
                        "invalid session id",
                        "no such session",
                        "session deleted",
                        "chrome not reachable",
                        "disconnected",
                    ])

                    if is_session_crash or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"    ⚠ Browser session lost — restarting Chrome...")
                        try:
                            driver.quit()
                        except Exception:
                            pass  # Already dead, ignore

                        _human_delay(4, 8)  # Brief pause before restarting
                        driver = _make_driver(headless=headless)
                        consecutive_failures = 0
                        print(f"    ✓ Browser restarted. Retrying match {match_id}...")

                        # Retry the same match with the fresh driver
                        try:
                            shots_df = scrape_match_shots(match_id, driver)
                            _append_shots(shots_df, shots_path)
                            new_shot_count += len(shots_df)
                            print(f"    ✓ {len(shots_df)} shots saved on retry")
                        except Exception as retry_exc:
                            print(f"    ⚠ Retry also failed, skipping match {match_id}: {retry_exc}")
                    else:
                        print(f"    ⚠ Skipped match {match_id}: {exc}")

                _human_delay(2.0, 6.0)
        finally:
            try:
                driver.quit()
            except Exception:
                pass

        print(f"\nAppended {new_shot_count} new shot rows → {shots_path}")
        results["shots_path"] = str(shots_path)

    print(f"\n✓ Run complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Understat match data (incremental).")
    parser.add_argument("--league",  default="EPL",  help="League code (default: EPL)")
    parser.add_argument("--season",  default=2024,   type=int, help="Season start year (default: 2024)")
    parser.add_argument("--shots",   action="store_true",      help="Scrape shot data for new matches only")
    parser.add_argument("--max",     default=None,   type=int, help="Max new matches to scrape per run")
    parser.add_argument("--visible", action="store_true",      help="Show the browser window (not headless)")
    args = parser.parse_args()

    pull_understat_data(
        league=args.league,
        season=args.season,
        include_shots=args.shots,
        max_matches=args.max,
        headless=not args.visible,
    )