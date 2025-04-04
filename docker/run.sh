#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

PLANTER=$SCRIPT_DIR/../
DATASETS=/mnt/data/datasets

CONTAINER_NAME="planter"

build() {
	pushd $SCRIPT_DIR >/dev/null
		docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) . -t $CONTAINER_NAME
	popd >/dev/null
}

run() {
	docker run \
		--rm \
		--privileged \
		--network host \
		-v $PLANTER:/home/docker/planter \
		-v $DATASETS:/home/docker/datasets \
		-v $SCRIPT_DIR/resources/.tmux.conf:/home/docker/.tmux.conf \
		-v $SCRIPT_DIR/resources/.vimrc:/home/docker/.vimrc \
		-v $SCRIPT_DIR/resources/.bashrc:/home/docker/.bashrc \
		-v $SCRIPT_DIR/resources/.inputrc:/home/docker/.inputrc \
		-v ~/.vim:/home/docker/.vim:ro \
		-v ~/.tmux:/home/docker/.tmux:ro \
		-v $HOME/.gitconfig:/home/docker/.gitconfig:ro \
		-v $HOME/.ssh:/home/docker/.ssh:ro \
		-v /var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket \
		-v ~/.Xauthority:/home/docker/.Xauthority \
		-it \
		$CONTAINER_NAME
}

build
run
