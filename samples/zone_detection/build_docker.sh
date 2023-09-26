#! /usr/bin/bash
TRAINING_DOCKER_IMAGE_NAME=tf1_od_api_env:1.0.0
docker build \
    --build-arg UBUNTU_VERSION=18.04 \
    --build-arg PIP=pip3 \
    --build-arg PYTHON=python3 \
    --build-arg USE_PYTHON_3_NOT_2=1 \
    --build-arg _PY_SUFFIX=3 \
    --build-arg TF_PACKAGE=tensorflow \
    --build-arg TF_PACKAGE_VERSION=1.15.5 \
    . -f Dockerfile \
    -t $TRAINING_DOCKER_IMAGE_NAME --network host