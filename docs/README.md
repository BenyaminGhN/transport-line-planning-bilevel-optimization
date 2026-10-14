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

## 2. Run the passenger-assignment example

```sh
python -m scripts.run_assignment
```

Expected output:

```text
Route flows: [60. 20.]
Unserved by OD: [20.]
Travel time: 900.0
Unserved penalty: 2000.0
Total objective: 2900.0
```

The example has 100 passengers and two predefined routes with travel times of
10 and 15 minutes. Both lines are enabled (`x=[1, 1]`), frequencies are `[3, 1]`
departures per hour, vehicle capacity is 20, and the planning horizon is one
hour. The routes carry 60 and 20 passengers; 20 remain unserved. With a penalty
of 100 per unserved passenger, the total objective is `900 + 2000 = 2900`.

To change the example, edit `example_instance()` for network, demand, line,
and route data, and `main()` for activation, frequencies, and penalty in
[scripts/run_assignment.py](../scripts/run_assignment.py).
The script currently takes no command-line arguments.

This script solves one lower-level passenger-assignment problem. The GA and
operator budget/fleet checks are not part of this example. See
[passenger_assignment.md](passenger_assignment.md) for the mathematical model
and integration notes.

## 3. Run tests

Run the implemented passenger-assignment tests:

```sh
python -m pytest tests/test_assignment_penalty.py -q
```

At the time of writing, this file contains 15 test cases, all passing in the
validated environment. For individual test names and outcomes, use:

```sh
python -m pytest tests/test_assignment_penalty.py -v
```

Run one test:

```sh
python -m pytest tests/test_assignment_penalty.py::test_capacity_shortage_is_penalized -v
```

Run the full suite and display reasons for skipped tests:

```sh
python -m pytest -q -rs
```

The last full-suite verification reported `31 passed, 17 skipped`. Existing
placeholder tests use `pytest.skip()`; skipped tests do not validate the
corresponding functionality. Counts will change as implementation progresses.
