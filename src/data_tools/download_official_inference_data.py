#!/usr/bin/env python3
"""Download official inference data packaged by the paper authors.

Downloads:
- distribution/*.parquet
- data/inference_data/**.parquet

Source repository:
https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from core.paths import OFFICIAL_DATA


API_TREE_URL = (
    "https://api.github.com/repos/Weixin-Liang/"
    "Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers/git/trees/main?recursive=1"
)
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/Weixin-Liang/"
    "Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers/main"
)
ROOT = OFFICIAL_DATA


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def wanted_path(path: str) -> bool:
    return (
        path.startswith("data/inference_data/")
        or path.startswith("distribution/")
    ) and path.endswith(".parquet")


def destination_for(path: str) -> Path:
    if path.startswith("data/"):
        return ROOT / path.removeprefix("data/")
    return ROOT / path


def download_one(path: str, size: int, *, retries: int = 4) -> str:
    destination = destination_for(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size == size:
        return "skip"

    url = f"{RAW_BASE_URL}/{path}"
    temp_destination = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                with temp_destination.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)

            if temp_destination.stat().st_size != size:
                raise OSError(
                    f"size mismatch: expected {size}, got {temp_destination.stat().st_size}"
                )
            temp_destination.replace(destination)
            return "download"
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {path}: {exc}") from exc
            time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to download {path}")


def main() -> None:
    tree = fetch_json(API_TREE_URL)
    files = [
        (item["path"], int(item["size"]))
        for item in tree.get("tree", [])
        if item.get("type") == "blob" and wanted_path(item.get("path", ""))
    ]
    files.sort()

    total_bytes = sum(size for _, size in files)
    print(f"Files to check: {len(files)}")
    print(f"Total packaged size: {total_bytes / 1024 / 1024:.2f} MiB")

    downloaded = 0
    skipped = 0
    downloaded_bytes = 0

    for index, (path, size) in enumerate(files, start=1):
        status = download_one(path, size)
        if status == "skip":
            skipped += 1
        else:
            downloaded += 1
            downloaded_bytes += size
        print(
            f"[{index:03d}/{len(files)}] {status:8} "
            f"{size / 1024 / 1024:7.2f} MiB  {path}",
            flush=True,
        )

    print()
    print(f"Downloaded files: {downloaded}")
    print(f"Skipped existing files: {skipped}")
    print(f"Downloaded this run: {downloaded_bytes / 1024 / 1024:.2f} MiB")
    print(f"Saved under: {ROOT}")


if __name__ == "__main__":
    main()
