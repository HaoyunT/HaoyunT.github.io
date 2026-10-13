#!/usr/bin/env python3
"""Scan a range of NeurIPS 2026 poster IDs for HaM-World-related papers.

The NeurIPS virtual site uses numeric poster URLs, so this script deliberately
walks every ID in the requested inclusive range. It uses only the Python
standard library and writes one JSON object per completed request, which makes
long scans resumable after a timeout or an interrupted process.

Example:
    python3 scripts/search_neurips_posters.py \
        --start 10000 --end 20000 \
        --checkpoint /tmp/neurips-2026-10k-20k.jsonl \
        --output /tmp/neurips-2026-hamworld.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://dev.neurips.cc/virtual/2026/poster/{id}"
DEFAULT_KEYWORDS = ("ham-world", "hamworld", "soft-hamiltonian")
USER_AGENT = "HaoyunT-homepage NeurIPS poster scanner/1.0"


class PosterParser(HTMLParser):
    """Extract the fields that are useful for matching a poster page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.active: list[tuple[str, int]] = []
        self.parts: dict[str, list[str]] = {
            "title": [],
            "event_title": [],
            "authors": [],
            "abstract": [],
        }

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        value = dict(attrs).get("class") or ""
        return set(value.split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = self._classes(attrs)
        field: str | None = None
        if tag == "title":
            field = "title"
        elif tag == "h1" and "event-title" in classes:
            field = "event_title"
        elif tag == "div" and "event-organizers" in classes:
            field = "authors"
        elif tag == "div" and "abstract-text-inner" in classes:
            field = "abstract"

        if field is not None:
            self.active.append((field, self.depth))
        self.depth += 1

    def handle_endtag(self, _tag: str) -> None:
        self.depth = max(0, self.depth - 1)
        self.active = [item for item in self.active if item[1] < self.depth]

    def handle_data(self, data: str) -> None:
        for field, _ in self.active:
            self.parts[field].append(data)

    def values(self) -> dict[str, str]:
        def clean(items: Iterable[str]) -> str:
            value = html.unescape(" ".join(items))
            return re.sub(r"\s+", " ", value).strip()

        return {key: clean(value) for key, value in self.parts.items()}


def parse_page(body: bytes) -> dict[str, str]:
    parser = PosterParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    values = parser.values()
    # The event title is more stable than the page title, but some pages may
    # omit the h1. Falling back keeps the scanner useful for those pages.
    if not values["event_title"]:
        values["event_title"] = values["title"]
    return values


def fetch_page(url: str, timeout: float, retries: int, max_bytes: int) -> tuple[int, bytes, str | None]:
    last_error: str | None = None
    for attempt in range(retries + 1):
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                body = response.read(max_bytes + 1)
                return status, body[:max_bytes], None
        except HTTPError as error:
            # A missing poster is a normal result during a brute-force scan.
            if error.code == 404:
                return 404, b"", None
            last_error = f"HTTP {error.code}"
        except (TimeoutError, URLError, OSError) as error:
            last_error = str(error)

        if attempt < retries:
            time.sleep(min(2.0, 0.35 * (2**attempt)))

    return 0, b"", last_error or "request failed"


def match_fields(values: dict[str, str], patterns: list[re.Pattern[str]]) -> list[str]:
    haystack = "\n".join(
        values.get(field, "") for field in ("event_title", "title", "authors", "abstract")
    )
    return [pattern.pattern for pattern in patterns if pattern.search(haystack)]


def scan_one(
    poster_id: int,
    base_url: str,
    timeout: float,
    retries: int,
    max_bytes: int,
    patterns: list[re.Pattern[str]],
    delay: float,
) -> dict[str, object]:
    if delay > 0:
        time.sleep(random.uniform(0, delay))

    url = base_url.format(id=poster_id)
    status, body, error = fetch_page(url, timeout, retries, max_bytes)
    result: dict[str, object] = {
        "id": poster_id,
        "url": url,
        "status": status,
        "matched": False,
    }
    if error is not None:
        result["error"] = error
        return result
    if status != 200:
        return result

    values = parse_page(body)
    matches = match_fields(values, patterns)
    result.update(
        {
            "title": values["event_title"],
            "page_title": values["title"],
            "authors": values["authors"],
            "abstract": values["abstract"],
            "matched": bool(matches),
            "matched_keywords": matches,
        }
    )
    return result


def load_completed(path: Path, retry_errors: bool) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                status = int(record.get("status", 0))
                if isinstance(record.get("id"), int) and (not retry_errors or status in (200, 404)):
                    completed.add(record["id"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    return completed


def build_patterns(keywords: Iterable[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for keyword in keywords:
        escaped = re.escape(keyword).replace(r"\-", r"[-\s]?")
        patterns.append(re.compile(escaped, re.IGNORECASE))
    return patterns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=10000, help="First poster ID (inclusive).")
    parser.add_argument("--end", type=int, default=20000, help="Last poster ID (inclusive).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL template containing {id}.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent requests (default: 8).")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries for transient failures.")
    parser.add_argument("--delay", type=float, default=0.1, help="Random delay before each request (seconds).")
    parser.add_argument("--max-bytes", type=int, default=256_000, help="Maximum HTML bytes read per page.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("neurips_poster_scan.jsonl"),
        help="JSONL checkpoint path (default: neurips_poster_scan.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurips_hamworld_matches.jsonl"),
        help="JSONL match output path (default: neurips_hamworld_matches.jsonl).",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        dest="keywords",
        help="Additional case-insensitive keyword; may be repeated.",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing checkpoint and start over.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Revisit IDs whose previous checkpoint record was an error.",
    )
    args = parser.parse_args()
    if args.start < 0 or args.end < args.start:
        parser.error("--end must be greater than or equal to --start, and IDs must be non-negative")
    if args.workers < 1 or args.retries < 0 or args.timeout <= 0 or args.delay < 0 or args.max_bytes < 1:
        parser.error("workers, timeout, retries, delay, and max-bytes must be valid positive values")
    return args


def main() -> int:
    args = parse_args()
    keywords = list(DEFAULT_KEYWORDS)
    if args.keywords:
        keywords.extend(args.keywords)
    patterns = build_patterns(keywords)

    completed = set() if args.fresh else load_completed(args.checkpoint, args.retry_errors)
    poster_ids = [poster_id for poster_id in range(args.start, args.end + 1) if poster_id not in completed]
    total = args.end - args.start + 1
    print(
        f"Scanning {len(poster_ids)} of {total} poster IDs "
        f"({args.start}-{args.end}) with {args.workers} workers...",
        flush=True,
    )
    if not poster_ids:
        print("Nothing to scan; checkpoint already covers this range.")
        return 0

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_handle = args.checkpoint.open("a", encoding="utf-8")
    output_handle = args.output.open("a", encoding="utf-8")
    lock = threading.Lock()
    matches = 0
    finished = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    scan_one,
                    poster_id,
                    args.base_url,
                    args.timeout,
                    args.retries,
                    args.max_bytes,
                    patterns,
                    args.delay,
                ): poster_id
                for poster_id in poster_ids
            }
            for future in as_completed(futures):
                record = future.result()
                with lock:
                    checkpoint_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoint_handle.flush()
                    if record.get("matched"):
                        output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output_handle.flush()
                        matches += 1
                    finished += 1
                if finished % 100 == 0 or finished == len(poster_ids):
                    print(f"Progress: {finished}/{len(poster_ids)}; matches={matches}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted. Completed requests remain in the checkpoint; rerun to resume.", file=sys.stderr)
        return 130
    finally:
        checkpoint_handle.close()
        output_handle.close()

    print(f"Done. Found {matches} matching poster(s). Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
