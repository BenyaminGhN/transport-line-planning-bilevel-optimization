"""Mandl benchmark: GA (and, with --exact, the exhaustive enumeration, about 2 minutes).   >>> CHANGED <<<

Run from the repository root:
    python -m scripts.run_mandl --viz save
"""
from scripts.common import parse_args, run_study

if __name__ == "__main__":
    run_study("mandl", parse_args(__doc__, default_exact=False))
