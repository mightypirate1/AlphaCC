hello:
	@echo "works?"

env-create:
	@bash make.sh env-create

env-delete:
	@bash make.sh env-delete

build:
	source .venv/bin/activate
	@bash make.sh build

env-activate:
	. .venv/bin/activate

test:
	@bash make.sh test

install:
	@bash make.sh install
	@echo "\n------------------------\nInstallation successful!\n------------------------"
