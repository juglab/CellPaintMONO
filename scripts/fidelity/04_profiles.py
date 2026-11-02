"""Step 4: well profile retrieval and distance geometry.

Usage:
    python scripts/fidelity/04_profiles.py --config configs/fidelity/<exp>.yaml
"""

from cellpaintmono.fidelity.cli import main

if __name__ == "__main__":
    main(["profiles"])
