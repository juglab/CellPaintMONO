"""Step 6: agreement and distortion per feature group (needs steps 2 and 3).

Usage:
    python scripts/fidelity/06_summary.py --config configs/fidelity/<exp>.yaml
"""

from cellpaintmono.fidelity.cli import main

if __name__ == "__main__":
    main(["summary"])
