#!/bin/sh

set -eu

docker build --tag docs:latest -f docker/dockerfile-docs .
docker cp "$(docker ps -alq):/home/docs/rocsolver/docs" built_docs
