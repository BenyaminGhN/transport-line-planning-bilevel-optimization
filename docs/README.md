# Environment setup and running the project

This guide covers Windows PowerShell, macOS, and Linux. Run project commands
from the repository root (the directory containing `pyproject.toml`).

Related documentation:

- [Passenger-assignment model](passenger_assignment.md)
- [Line planning and solution search](line_planning.md)

## Prerequisites

- Python 3.10 or newer, with `pip` and `venv` available.

The passenger-assignment example has been tested with Python 3.12. Dependencies
are declared in `pyproject.toml`; their versions are currently not pinned.

## 1. Create a virtual environment

Each contributor should create their own local `.venv`. The directory is
excluded from version control; do not copy it between computers or commit it.
Use a standard virtual environment without `--system-site-packages` to keep
project dependencies separate from globally installed packages.

### Windows (PowerShell)

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If `python` is not available but the Windows Python launcher is installed,
use `py -3 --version` and `py -3 -m venv .venv` for the first two commands.
Check that the selected version meets the requirement above.

### Verify the installation

With the environment activated, run:

```sh
python -c "import sys, ptlp; print(sys.executable); print(ptlp.__file__)"
```

The interpreter should be inside the project's `.venv`, and the package path
should point to `src/ptlp/__init__.py` in your checkout.

`pip install -e .` installs the project in editable mode, together with its
declared dependencies. Source edits are available on the next run without
reinstalling. Run the installation command again if dependencies change.

## 2. Quick start (5 commands)

> 🔶 **CHANGED**: this section is new. The old "run one assignment" example still works (`python -m scripts.run_assignment`).

```sh
python -m scripts.generate_data      # writes data/synthetic/* and the candidate lines of data/mandl/  (1 second)
python -m scripts.run_synthetic      # exact answer + GA on the synthetic city, with GIF   (about 40 seconds)
python -m scripts.run_baselines      # compares with the two baselines of the proposal     (about 30 seconds)
python -m scripts.run_sensitivity    # capacity, weights, omega, budget, fleet, routes      (about 90 seconds)
python -m scripts.run_mandl          # GA on the Mandl network (add --exact for the exact answer, about 2 minutes)
```

Everything is written to `results/`. Useful flags (all scripts): `--viz off|live|save|both`, `--runs`, `--seed`,
`--pop`, `--gens`, `--lambda-op`, `--uncapacitated`. Run `python -m scripts.run_synthetic --help` to see them.

`--viz live` opens a window that is redrawn after every generation (needs a normal desktop Python, not a notebook
or a server). `--viz save` writes `results/<name>/frames/frame_XXXX.png` and `results/<name>/ga.gif`.

## 3. Run the tests

```sh
python -m pytest -q          # 71 tests, about 20 seconds
```

## 4. Where to read next

| You want to know | Read |
|---|---|
| What the passengers' LP does | [passenger_assignment.md](passenger_assignment.md) |
| How the line plan is searched (GA, exact, tie-breaking) | [line_planning.md](line_planning.md) |
| Where the data comes from and what each experiment shows | [data_and_experiments.md](data_and_experiments.md) |
