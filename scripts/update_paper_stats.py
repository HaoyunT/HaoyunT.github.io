#!/usr/bin/env python3
"""Refresh the metric counts embedded in index.html.

Two kinds of count live in the markup:

* alphaXiv like counts, from api.alphaxiv.org, which sends no CORS headers for
  third-party origins and therefore cannot be fetched by the browser.
* GitHub star counts, from api.github.com, which does allow CORS but rate-limits
  unauthenticated callers to 60 requests per hour per IP; shared-NAT visitors
  routinely exhaust that and silently fall back to the baked-in number.

Both run server-side in GitHub Actions, where there is no same-origin policy and
where an optional GITHUB_TOKEN raises the API limit from 60 to 5000 per hour.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ALPHAXIV_TEMPLATE = "https://api.alphaxiv.org/papers/v3/legacy/{paper_id}"
GITHUB_TEMPLATE = "https://api.github.com/repos/{repository}"
INDEX_PATH = pathlib.Path(__file__).resolve().parent.parent / "index.html"
REQUEST_TIMEOUT = 30

ALPHAXIV_ANCHOR_RE = re.compile(
    r'<a\b[^>]*data-alphaxiv-paper="(?P<paper>[^"]+)"[^>]*>.*?</a>',
    re.DOTALL,
)
ALPHAXIV_COUNT_RE = re.compile(r'(<span class="alphaxiv-like-count">)(\d+)(</span>)')

GITHUB_ANCHOR_RE = re.compile(
    r'<a\b[^>]*data-github-repo="(?P<repo>[^"]+)"[^>]*>.*?</a>',
    re.DOTALL,
)
GITHUB_LABEL_RE = re.compile(r'(aria-label|title)="\d+ GitHub stars?"')
GITHUB_COUNT_RE = re.compile(r'(<span class="repo-star-count">)\d+(</span>)')


def _get_json(url: str, extra_headers: dict[str, str] | None = None) -> object:
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; HaoyunTangHomepageBot/1.0; +https://haoyunt.github.io/)",
        "Origin": "https://haoyunt.github.io",
        "Referer": "https://haoyunt.github.io/",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.load(response)


def _as_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def fetch_like_count(paper_id: str) -> int | None:
    url = ALPHAXIV_TEMPLATE.format(paper_id=urllib.parse.quote(paper_id, safe=""))
    payload = _get_json(url)
    # The payload shape is not contractual: `paper` can be null or a list, and
    # any level can be missing. Walk it defensively so one odd response skips a
    # paper instead of aborting the whole run.
    paper = payload.get("paper") if isinstance(payload, dict) else None
    paper_group = paper.get("paper_group") if isinstance(paper, dict) else None
    metrics = paper_group.get("metrics") if isinstance(paper_group, dict) else None
    value = metrics.get("public_total_votes") if isinstance(metrics, dict) else None
    return _as_count(value)


def fetch_star_count(repository: str) -> int | None:
    url = GITHUB_TEMPLATE.format(repository=repository)
    payload = _get_json(url, {"Accept": "application/vnd.github+json"})
    stars = payload.get("stargazers_count") if isinstance(payload, dict) else None
    return _as_count(stars)


def update_alphaxiv_anchor(match: re.Match) -> str:
    original = match.group(0)
    paper_id = match.group("paper")
    try:
        likes = fetch_like_count(paper_id)
    except (urllib.error.URLError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
        print(f"  alphaxiv {paper_id}: skipped ({exc})", file=sys.stderr)
        return original
    if likes is None:
        print(f"  alphaxiv {paper_id}: no public_total_votes in payload", file=sys.stderr)
        return original

    label = f"{likes} like{'s' if likes != 1 else ''} on alphaXiv"
    updated = re.sub(r'aria-label="[^"]*"', f'aria-label="{label}"', original, count=1)
    updated = re.sub(r'title="[^"]*"', f'title="{label}"', updated, count=1)
    updated = ALPHAXIV_COUNT_RE.sub(lambda m: f"{m.group(1)}{likes}{m.group(3)}", updated, count=1)
    print(f"  alphaxiv {paper_id}: {likes}")
    return updated


def update_github_anchor(match: re.Match) -> str:
    original = match.group(0)
    repository = match.group("repo")
    try:
        stars = fetch_star_count(repository)
    except (urllib.error.URLError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
        print(f"  github {repository}: skipped ({exc})", file=sys.stderr)
        return original
    if stars is None:
        print(f"  github {repository}: no stargazers_count in payload", file=sys.stderr)
        return original

    label = f"{stars} GitHub star{'s' if stars != 1 else ''}"
    updated = GITHUB_LABEL_RE.sub(lambda m: f'{m.group(1)}="{label}"', original)
    updated = GITHUB_COUNT_RE.sub(lambda m: f"{m.group(1)}{stars}{m.group(2)}", updated, count=1)
    print(f"  github {repository}: {stars}")
    return updated


def main() -> int:
    html = INDEX_PATH.read_text(encoding="utf-8")
    updated = ALPHAXIV_ANCHOR_RE.sub(update_alphaxiv_anchor, html)
    updated = GITHUB_ANCHOR_RE.sub(update_github_anchor, updated)

    if updated == html:
        print("index.html already up to date")
    else:
        INDEX_PATH.write_text(updated, encoding="utf-8")
        print("index.html updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
