"""Step 1: object counts, centroid matching and sizes of unmatched objects.

Usage:
    python scripts/fidelity/01_segmentation.py --config configs/fidelity/<exp>.yaml
"""

from cellpaintmono.fidelity.cli import main

if __name__ == "__main__":
    main(["segmentation"])
