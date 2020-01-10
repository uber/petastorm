all: rm-image build

# Build the Petastorm container
# Note - the build process can be memory intensive due to
# pip. If the install process is killed midway through,
# rerun it and check `docker stats` during execution; you may
# need to increase Docker's memory limits.
TAG := $(shell date +%Y-%m-%d_%H-%M-%S)
IMAGE := "docker.io/selitvin/petastorm_ci"
build:
	cd .. && docker build . \
	-f ./docker/Dockerfile \
	--tag $(IMAGE):$(TAG) --tag $(IMAGE):latest
	@echo
	@echo "You can push this image to docker.io by:"
	@echo
	@echo " 1. Log into docker.io:"
	@echo
	@echo "    docker login docker.io"
	@echo
	@echo " 2. Push the docker image to docker.io:"
	@echo
	@echo "    docker push $(IMAGE):$(TAG)"

# Run a container to serve as a development workspace with
# the local repo mounted into it. IMPORTANT: this gives
# the container access to the host's filesystem!
run:
	docker run \
	-it \
	-v `pwd`/..:/petastorm \
	$(IMAGE):latest

# Expunge old container image
rm-image:
	-docker rmi $(IMAGE)
