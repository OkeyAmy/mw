PY ?= .venv/bin/python
export PYTHONPATH := code

MODEL ?= .hf/bf16-hf
DTYPE ?= float32

.PHONY: help install fast slow lint check prove clean backend frontend up

help:
	@echo "make install  - create venv and install deps"
	@echo "make fast     - tests that need no model (CI gate)"
	@echo "make slow     - tests that load real weights (release gate)"
	@echo "make lint     - ruff; FAILS on findings (no || true)"
	@echo "make check    - fast + lint"
	@echo "make prove    - run the full proof chain"

install:
	uv venv --python 3.12 .venv
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m pip install ruff

fast:
	$(PY) -m pytest tests -m "not slow"

slow:
	$(PY) -m pytest tests -m "slow"

lint:
	$(PY) -m ruff check code app tests

# G1 D2: `ruff ... || true` made lint unable to fail. It is now a real gate.
check: lint fast

prove:
	$(PY) code/proof_chain.py

# --- live app -------------------------------------------------------------
backend:
	PYTHONPATH=code $(PY) app/server.py --model $(MODEL) --dtype $(DTYPE)

frontend:
	cd web && npm run dev

# starts the backend in the background, then the frontend in the foreground
up:
	@echo "backend -> http://127.0.0.1:8000   frontend -> http://127.0.0.1:3000"
	@PYTHONPATH=code $(PY) app/server.py --model $(MODEL) --dtype $(DTYPE) & \
	 sleep 3; cd web && npm run dev

clean:
	rm -rf proof/*.npy proof/*.pt .pytest_cache
