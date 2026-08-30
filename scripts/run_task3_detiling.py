"""Map task 3 tile predictions back to page coordinates.

context-stripped model, held-out test split.

Run from the repository root:  python scripts/run_task3_detiling.py
"""

from scribal_char_spotting.tiling import run_detiling_for_task

if __name__ == "__main__":
    run_detiling_for_task(3)
