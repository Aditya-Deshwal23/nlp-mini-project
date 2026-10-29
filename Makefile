PYTHON ?= python3.11

.PHONY: setup test lint format check clean

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/python -m spacy download en_core_web_sm

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .

check: lint
	.venv/bin/coverage run -m pytest
	.venv/bin/coverage report --fail-under=85
	.venv/bin/python scripts/validate_docs.py
	.venv/bin/python scripts/check_svg.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
