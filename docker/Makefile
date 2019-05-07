CONTAINER_NAME:=petastorm-test
IMAGE_NAME:=petastorm

all: rm-image build

# Build the Petastorm container
# Note - the build process can be memory intensive due to
# pip. If the install process is killed midway through,
# rerun it and check `docker stats` during execution; you may
# need to increase Docker's memory limits.
build:
	cd .. && docker build . \
	-f ./docker/Dockerfile \
	--tag $(IMAGE_NAME)

# Create the CI container for Travis use
push_ci_container:
	docker tag $(IMAGE_NAME):latest selitvin/petastorm_ci:latest
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin
	docker push selitvin/petastorm_ci:latest

# Run a container to serve as a development workspace with
# the local repo mounted into it. IMPORTANT: this gives
# the container access to the host's filesystem!
run:
	docker run \
	-dt \
	-v `pwd`/..:/petastorm \
	--name $(CONTAINER_NAME) \
	$(IMAGE_NAME)

# Create a shell into a running container
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Stop and remove running containers
clean:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

# Expunge old container image
rm-image:
	-docker rmi $(IMAGE_NAME)
