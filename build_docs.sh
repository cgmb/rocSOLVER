#!/bin/sh

set -eu

docker build --tag docs:latest -f docker/dockerfile-docs .
container_id="$(docker create docs:latest)"
docker cp "$container_id:/home/docs/rocsolver/docs" built_docs
