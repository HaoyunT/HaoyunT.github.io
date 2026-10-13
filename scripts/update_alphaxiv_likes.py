#!/usr/bin/env python3
"""Refresh alphaXiv like counts embedded in index.html.

alphaXiv's public API does not send CORS headers for third-party origins, so
the counts cannot be fetched by the browser. This runs server-side in GitHub
Actions, where the same-origin policy does not apply.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

API_TEMPLATE = "https://api.alphaxiv.org/papers/v3/legacy/{paper_id}"
INDEX_PATH = pathlib.Path(__file__).resolve().parent.parent / "index.html"
REQUEST_TIMEOUT = 30

ANCHOR_RE = re.compile(
    r'<a\b[^>]*data-alphaxiv-paper="(?P<paper>[^"]+)"[^>]*>.*?</a>',
    re.DOTALL,
)
COUNT_RE = re.compile(r'(<span class="alphaxiv-like-count">)(\d+)(</span>)')


def fetch_like_count(paper_id: str) -> int | None:
    url = API_TEMPLATE.format(paper_id=urllib.parse.quote(paper_id, safe=""))
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; HaoyunTangHomepageBot/1.0; +https://haoyunt.github.io/)",
            "Origin": "https://haoyunt.github.io",
            "Referer": "https://haoyunt.github.io/",
        },
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        payload = json.load(response)
    # The payload shape is not contractual: `paper` can be null or a list, and
    # any level can be missing. Walk it defensively so one odd response skips a
    # paper instead of aborting the whole run.
    paper = payload.get("paper") if isinstance(payload, dict) else None
    paper_group = paper.get("paper_group") if isinstance(paper, dict) else None
    metrics = paper_group.get("metrics") if isinstance(paper_group, dict) else None
    value = metrics.get("public_total_votes") if isinstance(metrics, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def update_anchor(match: re.Match) -> str:
    original = match.group(0)
    paper_id = match.group("paper")
    try:
        likes = fetch_like_count(paper_id)
    except (urllib.error.URLError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
        print(f"  {paper_id}: skipped ({exc})", file=sys.stderr)
        return original
    if likes is None:
        print(f"  {paper_id}: no public_total_votes in payload", file=sys.stderr)
        return original

    label = f"{likes} like{'s' if likes != 1 else ''} on alphaXiv"
    updated = re.sub(r'aria-label="[^"]*"', f'aria-label="{label}"', original, count=1)
    updated = re.sub(r'title="[^"]*"', f'title="{label}"', updated, count=1)
    updated = COUNT_RE.sub(lambda m: f"{m.group(1)}{likes}{m.group(3)}", updated, count=1)
    print(f"  {paper_id}: {likes}")
    return updated


def main() -> int:
    html = INDEX_PATH.read_text(encoding="utf-8")
    updated = ANCHOR_RE.sub(update_anchor, html)
    if updated == html:
        print("index.html already up to date")
    else:
        INDEX_PATH.write_text(updated, encoding="utf-8")
        print("index.html updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
