PYTHON ?= python3

# GPU tests run only where torch sees a GPU ("everything that runs on this machine"; CUDA_VISIBLE_DEVICES respected,
# D-020e). Recursive (=) so that only `make test` pays for importing torch.
HAVE_GPU = $(shell $(PYTHON) -c "import sys, torch; sys.exit(not torch.cuda.is_available())" >/dev/null 2>&1 && echo 1)

.PHONY: setup fetch lint test-fast test smoke

setup:
	$(PYTHON) -m pip install -e ".[dev,train]"

# The pinned tokenizers (which the tests read) and the smoke model, into the Hugging Face cache. Needs network and an
# account with access to the gated Llama repositories; run once, on the login node.
fetch:
	$(PYTHON) scripts/fetch_models.py --weights Qwen/Qwen2.5-0.5B-Instruct
	$(PYTHON) scripts/fetch_toolhop.py

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

test-fast:
	$(PYTHON) -m pytest tests reference -q -m "not gpu and not slow"

test:
	$(PYTHON) -m pytest tests reference -q $(if $(HAVE_GPU),,-m "not gpu")

# Configurations 2 and 6 on the toy env, evaluation and a table; one GPU (qsub -o logs/pbs/ scripts/pbs/smoke.pbs).
smoke:
	$(PYTHON) scripts/smoke.py
