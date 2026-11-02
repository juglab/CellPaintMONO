"""Step 5: loss of cell-to-cell spread and intensity shifts.

Usage:
    python scripts/fidelity/05_spread.py --config configs/fidelity/<exp>.yaml
"""

from cellpaintmono.fidelity.cli import main

if __name__ == "__main__":
    main(["spread"])
