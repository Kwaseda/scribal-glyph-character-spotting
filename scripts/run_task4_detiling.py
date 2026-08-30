"""Map task 4 tile predictions back to page coordinates.

context-stripped model over the blanked training tiles.

Run from the repository root:  python scripts/run_task4_detiling.py
"""

from scribal_char_spotting.tiling import run_detiling_for_task

if __name__ == "__main__":
    run_detiling_for_task(4)
