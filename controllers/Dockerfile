FROM cyberbotics/webots.cloud:R2023a-ubuntu20.04

# Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -yq --no-install-recommends \
        git \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Environment variables needed for Webots
# https://cyberbotics.com/doc/guide/running-extern-robot-controllers#remote-extern-controllers
ENV PYTHONPATH=${WEBOTS_HOME}/lib/controller/python
ENV PYTHONIOENCODING=UTF-8
ENV LD_LIBRARY_PATH=${WEBOTS_HOME}/lib/controller:${LD_LIBRARY_PATH}
ARG WEBOTS_CONTROLLER_URL
ENV WEBOTS_CONTROLLER_URL=${WEBOTS_CONTROLLER_URL}

# Environment variables needed for NVIDIA GPU
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="all"

# Prepares the controller of the participant
COPY .docker/scripts/setup.bash /usr/local/webots-project/controllers/.docker/scripts/setup.bash
RUN /usr/local/webots-project/controllers/.docker/scripts/setup.bash

# Downloads trained models
ARG MODELS_VERSION=0.3.2
COPY .docker/scripts/download_models.bash /usr/local/webots-project/controllers/.docker/scripts/download_models.bash
RUN /usr/local/webots-project/controllers/.docker/scripts/download_models.bash

# Copies all the files of the controllers folder into the docker container
COPY . /usr/local/webots-project/controllers

# Entrypoint command to launch default Python controller script
# (In shell form to allow variable expansion)
WORKDIR /usr/local/webots-project/controllers/participant
ENTRYPOINT python3 -u participant.py
