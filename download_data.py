"""
download_data.py
----------------
Downloads the NASA C-MAPSS dataset.
Tries two public GitHub mirrors. If both fail, prints manual instructions.

Usage:
    python download_data.py
"""

import os
import urllib.request

# Two confirmed-working mirrors (tested May 2026)
MIRRORS = [
    "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData",
    "https://raw.githubusercontent.com/DemetraAS/C-MAPSS/main",
]

FILES = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
]

SAVE_DIR = "data/raw"


def try_download(fname: str, dest: str) -> bool:
    """Try each mirror in order, return True on first success."""
    for mirror in MIRRORS:
        url = f"{mirror}/{fname}"
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = os.path.getsize(dest) // 1024
            print(f"OK  ({size_kb} KB)")
            return True
        except Exception:
            continue   # try next mirror
    return False


def download():
    os.makedirs(SAVE_DIR, exist_ok=True)
    success, skipped, failed = 0, 0, 0

    for fname in FILES:
        dest = os.path.join(SAVE_DIR, fname)

        if os.path.exists(dest) and os.path.getsize(dest) > 500:
            print(f"  [SKIP]     {fname}  (already exists)")
            skipped += 1
            continue

        print(f"  [DOWNLOAD] {fname} ...", end=" ", flush=True)
        if try_download(fname, dest):
            success += 1
        else:
            print("FAILED (all mirrors)")
            # clean up empty file if created
            if os.path.exists(dest) and os.path.getsize(dest) == 0:
                os.remove(dest)
            failed += 1

    print(f"\n  Downloaded: {success}  |  Skipped: {skipped}  |  Failed: {failed}")

    if failed > 0:
        print("\n  Automatic download failed. Get the data manually:")
        print("  Option A — Kaggle (free account needed):")
        print("    https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        print("    Download zip → extract → copy all .txt files to:  data/raw/")
        print()
        print("  Option B — NASA directly:")
        print("    https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
        print("    Download 'Turbofan Engine Degradation Simulation Data Set'")
    else:
        print(f"\n  All {success + skipped} files ready in data/raw/")
        print("  Next step:  python run_pipeline.py --dataset FD001")


if __name__ == "__main__":
    print("=== Downloading NASA C-MAPSS Dataset ===\n")
    download()
