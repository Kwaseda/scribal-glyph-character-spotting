"""Map task 2 tile predictions back to page coordinates.

unmodified tiles, held-out test split (baseline).

Run from the repository root:  python scripts/run_task2_detiling.py
"""

from scribal_char_spotting.tiling import run_detiling_for_task

if __name__ == "__main__":
    run_detiling_for_task(2)
