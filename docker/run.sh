#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

PLANTER=$SCRIPT_DIR/../
DATASETS=/mnt/data/datasets

SDE_VERSION="9.13.4"

CONTAINER_NAME="planter"
SDE="$SCRIPT_DIR/resources/bf-sde-$SDE_VERSION.tgz"
BSP="$SCRIPT_DIR/resources/bf-reference-bsp-$SDE_VERSION.tgz"

check_files() {
	if [ ! -f $SDE ]; then
		echo "Error: Missing SDE ($SDE). Exiting."
		exit 1
	fi

	if [ ! -f $BSP ]; then
		echo "Error: Missing bsp ($BSP). Exiting."
		exit 1
	fi
}

build() {
	pushd $SCRIPT_DIR >/dev/null
		docker build --build-arg USER_UID=$(id -u) . -t $CONTAINER_NAME
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
		-v ~/.Xauthority:/home/user/.Xauthority \
		-it \
		$CONTAINER_NAME
}

check_files
build
run
