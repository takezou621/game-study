.PHONY: deps deps-update deps-check install dev test lint format clean

deps:
	pip install -r requirements.txt

deps-update:
	pip-compile --generate-hashes requirements.in -o requirements.txt
	pip-compile --generate-hashes requirements-dev.in -o requirements-dev.txt

deps-check:
	pip-audit -r requirements.txt
	safety check -r requirements.txt

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest --cov=src --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	isort src tests

clean:
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
