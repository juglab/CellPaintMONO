"""Step 3: per-image distances against the noise floor and the well-to-well yardstick.

Usage:
    python scripts/fidelity/03_distances.py --config configs/fidelity/<exp>.yaml
"""

from cellpaintmono.fidelity.cli import main

if __name__ == "__main__":
    main(["distances"])
