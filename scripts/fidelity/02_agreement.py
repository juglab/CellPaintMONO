"""Step 2: per-cell agreement (Spearman rho) of matched objects.

Usage:
    python scripts/fidelity/02_agreement.py --config configs/fidelity/<exp>.yaml
"""

from cellpaintmono.fidelity.cli import main

if __name__ == "__main__":
    main(["agreement"])
