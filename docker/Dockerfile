FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Lisbon

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install -y sudo

ARG UID
ARG GID

RUN adduser --disabled-password --gecos '' --uid $UID docker && \
	adduser docker sudo && \
    echo 'docker ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers

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
