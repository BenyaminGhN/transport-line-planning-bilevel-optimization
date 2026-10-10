"""Synthetic study: exact enumeration + GA with per-generation visualisation.
Usage: python scripts/run_synthetic.py --viz save --seed 0"""
from __future__ import annotations
import argparse


def parse_args() -> argparse.Namespace:
    """Flags: --viz {off,live,save,both}  --seed  --pop  --gens  --lambda-op  --uncapacitated"""
    raise NotImplementedError


def main() -> None:
    """load_instance -> Evaluator -> enumerate_all -> GeneticAlgorithm(on_generation=LiveDashboard)
    -> save results/*.csv, final plots, gif, and print GA gap vs exact."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
