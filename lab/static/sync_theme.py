"""Copy lab/static/theme.css into each page, between its @theme markers.

The lab server only serves the three pages (index, constructor, catalog), so the theme can't be a linked file
without a server change. theme.css stays the single source of truth; this script keeps the inlined copies equal.

    python3 lab/static/sync_theme.py          # write
    python3 lab/static/sync_theme.py --check  # exit 1 if any page is out of date (for tests)
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = ["index.html", "constructor.html", "catalog.html"]
BEGIN, END = "/* @theme:begin (generated from theme.css by sync_theme.py; do not edit here) */", "/* @theme:end */"


def main(check: bool) -> int:
    theme = (HERE / "theme.css").read_text().strip()
    block = f"{BEGIN}\n{theme}\n{END}"
    stale = []
    for name in PAGES:
        path = HERE / name
        html = path.read_text()
        pat = re.compile(re.escape(BEGIN.split(" (")[0]) + r".*?" + re.escape(END), re.S)
        if pat.search(html):
            new = pat.sub(lambda _: block, html, count=1)
        else:  # first run: put the theme at the top of the page's first <style>
            new = html.replace("<style>", "<style>\n" + block + "\n", 1)
        if new != html:
            stale.append(name)
            if not check:
                path.write_text(new)
    if check:
        print("out of date: " + ", ".join(stale) if stale else "all pages carry the current theme")
        return 1 if stale else 0
    print("updated: " + (", ".join(stale) or "nothing (already current)"))
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
