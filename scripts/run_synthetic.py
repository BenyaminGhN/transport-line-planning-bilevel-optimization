"""Synthetic study: exact enumeration + GA with per-generation visualisation.   >>> CHANGED: implemented <<<

Run from the repository root:
    python -m scripts.run_synthetic --viz save --seed 0
Results (CSV, best plan picture, GA animation) go to results/synthetic/.
"""
from scripts.common import parse_args, run_study

if __name__ == "__main__":
    run_study("synthetic", parse_args(__doc__, default_exact=True))
