PYTHON ?= python3

# GPU tests run only where a GPU is visible ("everything that runs on this machine").
HAVE_GPU := $(shell nvidia-smi -L >/dev/null 2>&1 && echo 1)

.PHONY: setup lint test-fast test smoke

setup:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

test-fast:
	$(PYTHON) -m pytest tests reference -q -m "not gpu and not slow"

test:
	$(PYTHON) -m pytest tests reference -q $(if $(HAVE_GPU),,-m "not gpu")

smoke:
	@echo "make smoke is not implemented yet (docs/MILESTONES.md E1.7)" >&2; exit 1
