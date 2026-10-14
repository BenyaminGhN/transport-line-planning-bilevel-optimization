"""Netherlands study (20 cities, your fake-data generator): GA (add --exact for the exhaustive benchmark; about 20 lines, so it takes very long).   >>> NEW FILE <<<

Run from the repository root:
    python -m scripts.run_netherlands --viz save
"""
from scripts.common import parse_args, run_study

if __name__ == "__main__":
    run_study("netherlands", parse_args(__doc__, default_exact=False))
