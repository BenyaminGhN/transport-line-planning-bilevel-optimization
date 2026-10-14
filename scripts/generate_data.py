"""Write data/synthetic/*, data/mandl/candidate_lines.csv and data/netherlands/* (+ params.json each).   >>> NEW FILE <<<

Run from the repository root:
    python -m scripts.generate_data
"""
from pathlib import Path

from ptlp.datagen import generate_mandl_lines, generate_synthetic, import_network


def main(root: Path = Path("data")) -> None:
    generate_synthetic(root / "synthetic")
    generate_mandl_lines(root / "mandl")
    import_network(root / "netherlands")   # >>> NEW <<<
    print("written:", root / "synthetic", "," , root / "mandl", "and", root / "netherlands")


if __name__ == "__main__":
    main()
