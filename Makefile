PYTHON3 = python3.13

clean: clean-build clean-pyc clean-cache clean-venv clean-webapp clean-engine  ## remove all build, test, coverage and Python artifacts

clean-build:
	@rm -rf build/
	@rm -rf dist/
	@rm -rf .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

clean-cache:
	@rm -f .coverage
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf .ruff_cache

clean-venv: ## remove venv
	@rm -rf .venv

clean-webapp:
	@rm -rf webapp/node_modules

clean-engine:
	@bash -c " \
		cd engine && \
		cargo clean \
	"

develop: clean venv
	@bash -c "\
		rustup component add clippy && \
		source .venv/bin/activate && \
        pip install uv && \
		uv pip install -e ".[all]" \
			--index-url https://pypi.org/simple \
			--extra-index-url https://download.pytorch.org/whl/cu128 \
			--index-strategy unsafe-best-match \
    "

install: develop build-engine install-webapp

build-engine:
	@bash -c " \
		source .venv/bin/activate && \
		cd engine/python && \
		maturin develop --release \
	"

generate-stubs:
	@bash -c " \
		source .venv/bin/activate && \
		cd engine && \
		LD_LIBRARY_PATH=\$$(python3 -c \"import sysconfig; print(sysconfig.get_config_var('LIBDIR'))\"):\$$LD_LIBRARY_PATH \
		cargo run -p alpha-cc-python --bin stub_gen \
	"

install-webapp:
	@bash -c " \
		cd webapp && \
		npm install --include=dev \
	"

lint: lint-py lint-rs lint-webapp

lint-py:
	@ruff check alpha_cc tests
	@black --check alpha_cc tests
	@mypy alpha_cc tests

lint-rs:
	@bash -c "cd engine && cargo clippy --workspace --exclude alpha-cc-python -- -D warnings"

lint-webapp:
	@bash -c "cd webapp && ng lint"

lint-fix:
	@ruff check --fix-only alpha_cc tests

reformat:
	@ruff check --select I,W --fix-only alpha_cc tests
	@black alpha_cc tests

test: ## run tests quickly with the default Python
	@pytest
	@bash -c "cd engine && cargo test --workspace --exclude alpha-cc-python"

venv:
	@$(PYTHON3) -m venv .venv --prompt alpha-cc

build-webapp:
	@docker compose -f docker-compose.webapp.yaml build

run-webapp:
	@docker compose -f docker-compose.webapp.yaml up

build-and-run-webapp: build-webapp run-webapp
