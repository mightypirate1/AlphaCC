DOCKER_IMAGE_NAME=alpha-cc

docker-dev-build:
	docker build -t $(DOCKER_IMAGE_NAME):dev  .

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME):latest  .

docker-debug:
	docker run \
		-it \
  	--name $(DOCKER_IMAGE_NAME)-devtest \
  	--mount type=bind,source=$(PWD)/agents,target=/AlphaCC/agents \
  	$(DOCKER_IMAGE_NAME):dev

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
