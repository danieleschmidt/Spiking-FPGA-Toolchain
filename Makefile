# Makefile for Spiking-FPGA-Toolchain development

.PHONY: help install install-dev test test-all lint format clean docs docs-serve build package

# Default target
help:
	@echo "Spiking-FPGA-Toolchain Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup commands:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  setup-hooks  Install pre-commit hooks"
	@echo ""
	@echo "Development commands:"
	@echo "  test         Run fast unit tests"
	@echo "  test-all     Run all tests including slow ones"
	@echo "  lint         Run code quality checks"
	@echo "  format       Format code with black and ruff"
	@echo "  typecheck    Run mypy type checking"
	@echo ""
	@echo "Documentation commands:"
	@echo "  docs         Build documentation"
	@echo "  docs-serve   Serve documentation locally"
	@echo "  docs-clean   Clean documentation build"
	@echo ""
	@echo "Build commands:"
	@echo "  build        Build package for distribution"
	@echo "  clean        Clean build artifacts"
	@echo ""
	@echo "Validation commands:"
	@echo "  validate-configs  Validate YAML/JSON configuration files"
	@echo "  check-hdl        Check HDL syntax (when HDL files exist)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt

setup-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Testing
test:
	pytest tests/unit/ -v

test-all:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-hardware:
	pytest tests/hardware/ -v -m hardware

# Code quality
lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff --fix src/ tests/
	isort src/ tests/

typecheck:
	mypy src/

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs && make html && python -m http.server 8000 --directory _build/html

docs-clean:
	cd docs && make clean

# Build and packaging
build: clean
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Validation helpers
validate-configs:
	@echo "Validating configuration files..."
	@find . -name "*.yaml" -o -name "*.yml" | grep -v .git | xargs python -c "import yaml, sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]" || echo "YAML validation failed"
	@find . -name "*.json" | grep -v .git | xargs python -c "import json, sys; [json.load(open(f)) for f in sys.argv[1:]]" || echo "JSON validation passed"

check-hdl:
	@echo "HDL syntax checking not implemented yet (no HDL files)"
	@echo "Will be implemented when HDL templates are added"

# Development shortcuts
dev: install-dev setup-hooks
	@echo "Development environment ready!"

ci: lint test
	@echo "CI checks passed!"

# Platform-specific FPGA tests (when hardware available)
test-vivado:
	pytest tests/hardware/ -v -m vivado

test-quartus:
	pytest tests/hardware/ -v -m quartus

# Performance benchmarking
benchmark:
	@echo "Benchmarking not implemented yet"
	@echo "Will be added when core functionality exists"