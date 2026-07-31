#!/bin/sh
# Rebuild the Linux CI image and publish it where the self-hosted runner reads it.
#
# The runner runs `docker pull` before creating a container job and fails if it
# cannot, even for an image already in the local daemon -- so a plain
# `docker build -t vkml-ci` is not enough and the image has to come from a
# registry. This one runs on the same host, so the "pull" is a loopback copy and
# there are no credentials to expire.
#
# Run after changing Dockerfile.ci. CI does NOT rebuild this itself: an image
# that rebuilds per run is an image whose contents can change without a commit,
# and the pinned toolchain is the entire point of having it.
set -eu

cd "$(dirname "$0")/.."

REGISTRY=localhost:5000
IMAGE="$REGISTRY/vkml-ci:latest"

if [ -z "$(docker ps -q -f name=vkml-registry)" ]; then
    echo "starting the local registry"
    docker run -d --restart=always -p 127.0.0.1:5000:5000 --name vkml-registry registry:2
fi

docker build -f Dockerfile.ci -t "$IMAGE" .
docker push "$IMAGE"

echo
echo "published $IMAGE"
docker run --rm --device /dev/dri:/dev/dri "$IMAGE" sh -c '
    printf "  python ........ %s\n" "$(python3 --version 2>&1)"
    printf "  clang-format .. %s\n" "$(clang-format --version)"
    printf "  vulkan devices  %s\n" "$(vulkaninfo --summary 2>/dev/null | grep -c deviceName)"
'
