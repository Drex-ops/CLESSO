#!/usr/bin/env python3
"""
Download geonpy .npy files from AARNet FileSender links.

Reads URLs from file_links.txt (same directory as this script) and saves
each file into the specified output directory, using the filename provided
by the server's Content-Disposition header.

Usage:
    python download_geonpy.py [--output-dir /path/to/output] [--links-file /path/to/file_links.txt]

Defaults:
    --output-dir   /Volumes/DATA/MAIN/NATIONAL/CLIMATE/geonpy
    --links-file   <script_dir>/file_links.txt
"""

import argparse
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def parse_filename_from_headers(response):
    """Extract filename from Content-Disposition header."""
    cd = response.headers.get("Content-Disposition", "")
    if cd:
        # Try quoted filename first: filename="some_file.npy"
        match = re.search(r'filename="(.+?)"', cd)
        if match:
            return match.group(1)
        # Try unquoted: filename=some_file.npy
        match = re.search(r"filename=(\S+)", cd)
        if match:
            return match.group(1)
    return None


def download_file(url, output_dir, index, total, max_retries=3):
    """Download a single file with retries and progress reporting."""
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as response:
                filename = parse_filename_from_headers(response)
                if not filename:
                    # Fallback: use the file_ids parameter as filename
                    match = re.search(r"files_ids=(\d+)", url)
                    fid = match.group(1) if match else str(index)
                    filename = f"file_{fid}.npy"

                dest = os.path.join(output_dir, filename)

                # Skip if already downloaded
                content_length = response.headers.get("Content-Length")
                if os.path.exists(dest) and content_length:
                    existing_size = os.path.getsize(dest)
                    if existing_size == int(content_length):
                        print(f"  [{index}/{total}] SKIP (exists): {filename}")
                        return filename, True

                # Download with progress
                total_size = int(content_length) if content_length else None
                downloaded = 0
                chunk_size = 1024 * 256  # 256 KB chunks

                size_str = f" ({total_size / 1024 / 1024:.1f} MB)" if total_size else ""
                print(f"  [{index}/{total}] Downloading: {filename}{size_str}")

                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size and sys.stdout.isatty():
                            pct = downloaded / total_size * 100
                            bar_len = 30
                            filled = int(bar_len * downloaded / total_size)
                            bar = "█" * filled + "░" * (bar_len - filled)
                            print(
                                f"\r    {bar} {pct:5.1f}% ({downloaded / 1024 / 1024:.1f} MB)",
                                end="",
                                flush=True,
                            )

                if total_size and sys.stdout.isatty():
                    print()  # newline after progress bar

                return filename, True

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"  [{index}/{total}] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [{index}/{total}] FAILED after {max_retries} attempts: {url}")
                return None, False


def main():
    script_dir = Path(__file__).resolve().parent
    default_links = script_dir / "file_links.txt"

    parser = argparse.ArgumentParser(description="Download geonpy files from AARNet FileSender")
    parser.add_argument(
        "--output-dir",
        default="/Volumes/DATA/MAIN/NATIONAL/CLIMATE/geonpy",
        help="Directory to save downloaded files (default: /Volumes/DATA/MAIN/NATIONAL/CLIMATE/geonpy)",
    )
    parser.add_argument(
        "--links-file",
        default=str(default_links),
        help=f"Path to file containing download URLs (default: {default_links})",
    )
    args = parser.parse_args()

    # Read URLs
    links_path = Path(args.links_file)
    if not links_path.exists():
        print(f"Error: Links file not found: {links_path}")
        sys.exit(1)

    urls = [line.strip() for line in links_path.read_text().splitlines() if line.strip()]
    print(f"Found {len(urls)} URLs in {links_path.name}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Download all files
    succeeded = 0
    failed = 0
    skipped = 0
    failed_urls = []

    for i, url in enumerate(urls, start=1):
        filename, ok = download_file(url, str(output_dir), i, len(urls))
        if ok:
            if filename and os.path.exists(os.path.join(str(output_dir), filename)):
                succeeded += 1
        else:
            failed += 1
            failed_urls.append(url)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Download complete: {succeeded} succeeded, {failed} failed out of {len(urls)} total")
    if failed_urls:
        print(f"\nFailed URLs:")
        for url in failed_urls:
            print(f"  {url}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
