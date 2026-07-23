# ─────────────────────────────────────────────────────────────────────────────
# Makefile  —  Convenience shortcuts for common tasks
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

PYTHON   = venv/bin/python
PIP      = venv/bin/pip
PYTEST   = venv/bin/pytest
FLAKE8   = venv/bin/flake8
BLACK    = venv/bin/black

.PHONY: venv install install-dev lint format test clean pipeline

# ── Environment setup ─────────────────────────────────────────────────────────
venv:
	python3 -m venv venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: venv
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	$(FLAKE8) src/ --max-line-length=120 --ignore=E402,W503,E501

format:
	$(BLACK) src/ run_pipeline.py --line-length 120

test:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

# ── Pipeline stages ───────────────────────────────────────────────────────────
stage1:
	PYTHONPATH=src $(PYTHON) src/prepare_data.py

stage2:
	PYTHONPATH=src $(PYTHON) src/infer_persona_llm.py

stage3a:
	PYTHONPATH=src $(PYTHON) src/generate_instructions.py

stage3b:
	PYTHONPATH=src $(PYTHON) src/build_instructions.py

stage4:
	PYTHONPATH=src $(PYTHON) src/fine_tune_lora.py

stage56:
	PYTHONPATH=src $(PYTHON) src/test_predict.py

stage7:
	PYTHONPATH=src $(PYTHON) src/evaluate.py

pipeline:
	PYTHONPATH=src $(PYTHON) run_pipeline.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/

clean-data:
	rm -rf data/processed/ data/personas/ instruction_dataset/

clean-outputs:
	rm -rf outputs/ models/ logs/

clean-all: clean clean-data clean-outputs
