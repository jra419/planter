FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Lisbon

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install -y sudo

RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%docker ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker
WORKDIR /home/docker

RUN mkdir /home/docker/.ssh
RUN chown -R docker:docker /home/docker/.ssh

RUN sudo apt update && sudo apt -y install \
	man \
	build-essential \
	wget \
	curl \
	git \
	vim \
	tzdata \
	tmux \
	iputils-ping \
	iproute2 \
	net-tools \
	tcpreplay \
	iperf \
	psmisc \
	htop \
	gdb \
	xdot \
	xdg-utils \
	python3 \
	python3-pip \
	fzf \
	clang \
	clang-tools \
	clangd \
	ripgrep \
	shellcheck \
	ninja-build \
	libpcap-dev \
	tcpdump \
	libomp-dev \
	libopenblas-base \
	libopenblas-dev \
	gfortran \
	libblas-dev \
	libatlas-base-dev \
	liblapack-dev \
	libjpeg8-dev \
	zlib1g-dev

RUN echo "wireshark-common wireshark-common/install-setuid boolean true" | sudo debconf-set-selections && \
	sudo apt install -y tshark

RUN cd /tmp/ && \
	wget https://go.dev/dl/go1.23.2.linux-amd64.tar.gz && \
	sudo tar -C /usr/local -xzf go1.23.2.linux-amd64.tar.gz && \
	echo "export PATH=$PATH:/usr/local" >> /home/docker/.profile && \
	echo "export PATH=$PATH:/usr/local/go/bin" >> /home/docker/.profile

ENV PATH="$PATH:/usr/local"
ENV PATH="$PATH:/usr/local/go/bin"

RUN git clone https://github.com/junegunn/fzf.git && \
	cd fzf && \
	awk 'NR==178 {$0="\tcp -f target/$(BINARY) /usr/bin/fzf"} 1' Makefile > Makefile.tmp && \
	rm Makefile && \
	mv Makefile.tmp Makefile && \
	make && \
	sudo PATH=$PATH:/usr/local:/usr/local/go/bin make install && \
	cd .. && \
	rm -r fzf

RUN sudo dpkg-reconfigure --frontend noninteractive tzdata

RUN echo "export PATH=/home/docker/.local/bin:\$PATH" >> /home/docker/.profile
ENV PATH="/home/docker/.local/bin:${PATH}"
RUN echo "source ~/.profile" >> /home/docker/.bash_profile

#############
#  SDE env	#
#############

ENV SDE=/home/docker/bf-sde-9.13.4
ENV SDE_INSTALL=/home/docker/bf-sde-9.13.4/install

COPY --chown=docker:docker resources/bf-sde-9.13.4.tgz .
COPY --chown=docker:docker resources/bf-reference-bsp-9.13.4.tgz .

RUN tar xvzf bf-sde-9.13.4.tgz

WORKDIR $SDE

RUN cd p4studio/dependencies && \
	awk 'NR==278 {$0="        attributes: {flags: --with-thread --with-test --with-filesystem --with-system install --with-graph --with-iostreams, url: 'https://archives.boost.io/release/1.67.0/source/boost_1_67_0.tar.bz2', version: 1.67.0}"} 1' dependencies.yaml > dependencies.yaml.tmp && \
	rm dependencies.yaml && \
	mv dependencies.yaml.tmp dependencies.yaml

RUN yes | ./p4studio/p4studio dependencies install

COPY --chown=docker:docker ./resources/profile.yaml .
RUN ./p4studio/p4studio profile apply profile.yaml

WORKDIR /home/docker

RUN echo "export SDE=/home/docker/bf-sde-9.13.4" >> ~/.profile
RUN echo "export SDE_INSTALL=/home/docker/bf-sde-9.13.4/install" >> ~/.profile
RUN echo "export PATH=/home/docker/bf-sde-9.13.4/install/bin:\$PATH" >> ~/.bash_profile

###################
#  Planter setup  #
###################

COPY --chown=docker:docker ./scripts/packages.txt /opt/packages.txt
RUN sudo pip3 install -r /opt/packages.txt

##########
#  Misc  #
##########

RUN pip3 install python-lsp-server
RUN pip3 install python-lsp-ruff

COPY --chown=docker:docker ./resources/alacritty.terminfo /home/docker/alacritty.terminfo
RUN tic -x /home/docker/alacritty.terminfo

CMD [ "/bin/bash", "-l"]
