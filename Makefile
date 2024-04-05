.PHONY: clean clean-build clean-pyc clean-cache clean-venv coverage develop install lint reformat template-update venv

PYTHON3 = python3.11

clean: clean-build clean-pyc clean-cache clean-venv  ## remove all build, test, coverage and Python artifacts

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-cache:
	rm -f .coverage
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache

clean-venv: ## remove venv
	rm -rf .venv

coverage: ## check code coverage quickly with the default Python
	coverage run --source alpha_cc -m pytest
	coverage report -m

develop: clean venv
	bash -c "\
	source .venv/bin/activate && \
	pip install -e .[dev] \
    "

install: develop build-engine

build-engine:
	bash -c "\
		source .venv/bin/activate && \
		cd alpha_cc/engine && \
		maturin develop \
	"

lint:
	ruff check alpha_cc tests
	black --check alpha_cc tests
	mypy alpha_cc tests

lint-fix:
	ruff check --fix-only alpha_cc tests

reformat:
	ruff check --select I,W --fix-only alpha_cc tests
	black alpha_cc tests

test: ## run tests quickly with the default Python
	pytest

template-update:
	cookiecutter_project_upgrader --context-file cookiecutter_input.json -p True -m True

venv:
	$(PYTHON3) -m venv .venv --prompt alpha-cc

github-init:
	git init --initial-branch=main
	git add .
	git commit -m "Generated from template"
	git remote add origin git@github.com:mightypirate1/alpha-cc.git
	git branch cookiecutter-template
	git push -u origin main
	git push -u origin cookiecutter-template