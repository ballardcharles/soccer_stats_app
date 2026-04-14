"""
Deep probe — shows raw script content so we can see the exact pattern understat uses.
Run: python probe2.py
"""
import re, requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

PAGES = {
    "league": "https://understat.com/league/EPL/2023",
    "player": "https://understat.com/player/2371",
    "team":   "https://understat.com/team/Manchester_City/2023",
}

for name, url in PAGES.items():
    r = requests.get(url, headers=HEADERS, timeout=30)
    print(f"\n{'='*60}")
    print(f"[{name}]  status={r.status_code}")

    soup = BeautifulSoup(r.text, "html.parser")
    scripts = soup.find_all("script")
    print(f"  <script> tags found: {len(scripts)}")

    for i, s in enumerate(scripts):
        text = (s.string or "").strip()
        if not text:
            continue
        print(f"\n  --- script[{i}] ({len(text)} chars) ---")
        # Show first 400 chars to see the pattern
        print("  " + text[:400].replace("\n", "\n  "))
        print("  ...")
        # Look for any JSON-like assignment pattern
        patterns = [
            r"JSON\.parse\((.{1,20})\)",       # what delimiter after JSON.parse(
            r"var\s+\w+\s*=\s*(.{1,30})",      # var assignments
        ]
        for pat in patterns:
            hits = re.findall(pat, text[:2000])
            if hits:
                print(f"  Pattern '{pat[:40]}' → {hits[:3]}")