"""
fetch_logos.py
--------------
One-time script to download Premier League club crests from the ESPN CDN
and cache them locally in assets/logos/.

Run once before launching the dashboard:
    python fetch_logos.py

Re-running is safe — already-downloaded files are skipped unless you pass
--force to overwrite them.
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Import team → ESPN ID mapping from the logos module
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from src.logos import TEAM_LOGO_IDS, ESPN_CDN

LOGOS_DIR = Path(__file__).parent / "assets" / "logos"
LOGOS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_all(force: bool = False) -> None:
    total   = len(TEAM_LOGO_IDS)
    skipped = 0
    downloaded = 0
    failed  = 0

    print(f"Downloading {total} club crests → {LOGOS_DIR}\n")

    for team, tid in sorted(TEAM_LOGO_IDS.items()):
        dest = LOGOS_DIR / f"{team}.png"

        if dest.exists() and not force:
            skipped += 1
            print(f"  ✓ {team:<25} (cached)")
            continue

        url = ESPN_CDN.format(id=tid)
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
            print(f"  ↓ {team:<25} {url}")
            time.sleep(0.15)   # polite rate limiting
        except Exception as exc:
            failed += 1
            print(f"  ✗ {team:<25} FAILED — {exc}")

    print(f"\nDone.  {downloaded} downloaded, {skipped} cached, {failed} failed.")
    if failed:
        print("Failed crests will show no image in the dashboard.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EPL club crests from ESPN CDN.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the file already exists.")
    args = parser.parse_args()
    fetch_all(force=args.force)
