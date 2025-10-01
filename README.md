# `risknyield`
A package for predicting crop yield at scale.

![Maize cumulative biomass evolution](examples/maize_biomass_evolution.gif)

---

## Introduction
`risknyield` implements a lightweight, transparent crop growth model. For a proper introduction, it's recommended to navigate locally the documentation. To that end, just open `docs/build/html/index.html` with your browser.

### Installation
#### Installation for Users
The straightforward way to install the package is to use the `.tar.gz` or `.whl` distribution files in the `dist/` directory. The easiest way to install the code is using `.whl` file, installing it in your virtual environment using `pip`:
```bash
pip install risknyield-*.whl
```

#### Install (developer mode)
For development however, it's better to clone the repo and make an editable install of the package in your virtual environment:
```bash
# while standing in the repo directory
pip install -e .
# optional dev tools (tests/docs/lint)
pip install -r dev-requirements.txt
```

### Quick example
Open **`examples/example_simulation.ipynb`** for a 5-minute tour of the model and how to run a simulation. Prefer a script? Run
```bash
python examples/run_simulation.py
```

---

## Repository layout (what’s in each folder)
- **`risknyield/`** – the Python package (source code)
  - `core/` – core data structures and the crop model (e.g., `CropModel`, `CropParams`, `Soil`, `Weather`, `Results`)
  - `library/` – reusable utilities (I/O, hydrology helpers, etc.)
- **`examples/`** – runnable notebooks and small scripts
  - `example_simulation.ipynb` – start here
  - `run_simulation.py` – script version of a basic run
  - `outputs/` – inputs and results of the examples
- **`data/`** – original data for maize 2021
  - `Soil/`, `Weather/` – original input files
  - `results/` – outputs
- **`tests/`** – regression tests (PyTest) and fixtures
- **`docs/`** – Sphinx documentation
  - `source/` – reStructuredText/Markdown sources
  - `build/` – generated HTML (open `build/html/index.html`)
- **`projects/`** – ad-hoc analyses / exploratory notebooks
- **`legacy/`** – historical materials (MATLAB code and PDFs)
- **Build & packaging files**
  - `pyproject.toml` – build config and dependencies
  - `MANIFEST.in` – packaging include rules
  - `dist/` – built wheels / source tarballs
  - `risknyield.egg-info/` – package metadata created during builds
- **Top-level configs**
  - `tox.ini` – test matrix configuration
  - `dev-requirements.txt` – tools for development (testing, docs, linting)
  - `LICENCE` – license text
  - `README.md` – this file

---

## Quality & standards

Aiming to keep the codebase simple, reliable, and easy to read, we follow industry standards for scientific computing. In particular:

* **Code Style:** Follows [PEP 8](https://peps.python.org/pep-0008/) (idiomatic names, spacing, imports).
* **Docstrings:** Follows [NumPy style for docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) with type hints; docs are built with Sphinx from the code.
* **Testing:** Pytest-based regression tests live in `tests/`. `tox` orchestrates running the test matrix and any auxiliary checks.

### Supported Python versions

`risknyield`'s behavior, packing, and installation are **tested on CPython 3.10, 3.11, and 3.12**. Other versions are not officially supported.

### Quick check with tox

```bash
# From the project root
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r dev-requirements.txt

tox                         # run the full matrix (e.g., py310, py311, py312)
# Useful variants:
tox -p all                  # run environments in parallel
tox -e py311                # run a single Python environment
tox -e docs                 # build documentation (if the docs env is defined)
```

## Tests

Run the full test suite:

```bash
pytest
```

Or run the matrix with `tox` (3.10 / 3.11 / 3.12, if you have them available):

```bash
tox
```

---

## Documentation

Sphinx docs live in **`docs/`**. To build locally:

```bash
cd docs/
make clean html
# then open docs/build/html/index.html
```

A separate page describing the mathematical model will be added to the docs.

---

## Status & versioning

Current pre-release: **0.0.1**. The API is still evolving, but core concepts (canopy cover, RUE, water and temperature stress responses, partitioning/harvest index) are stable.

---

## License

See **`LICENCE`**.