"""Run all feature fidelity steps (1 to 6) on one reference and test run.

Usage:
    python scripts/fidelity/run_all.py --config configs/fidelity/<exp>.yaml
"""

from cellpaintmono.fidelity.cli import main

if __name__ == "__main__":
    main()
