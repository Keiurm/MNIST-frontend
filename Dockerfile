# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:20.04

ARG PIP_INS_OPTS
ARG PYTHONWARNINGS
ARG CURL_OPTS
ARG WGET_OPTS
ARG APT_OPTS=true

ARG PYTHON_VERSION_MAJOR=3
ARG PYTHON_VERSION_MINOR=9
ENV PYVERNAME=${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}

RUN eval ${APT_OPTS} && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       software-properties-common \
       bzip2 \
       ca-certificates \
       curl \
       libglib2.0-0 \
       libgl1-mesa-glx \
       apache2 \
       libapache2-mod-wsgi-py3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

################################################## build python from pyenv
ARG WHL_PATH=/tmp
#ADD $WHL_PATH/*.whl /tmp/

RUN eval ${APT_OPTS} \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       git \
       make \
       build-essential \
       libssl-dev \
       zlib1g-dev \
       libbz2-dev \
       libreadline-dev \
       libsqlite3-dev \
       wget \
       llvm \
       libncursesw5-dev \
       xz-utils \
       tk-dev \
       libxml2-dev \
       libxmlsec1-dev \
       libffi-dev \
       liblzma-dev \
    && git clone https://github.com/pyenv/pyenv.git ~/.pyenv \
    && export PYENV_ROOT="$HOME/.pyenv" \
    && export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/plugins/python-build/bin:$PATH" \
    && export PYTHON_BUILD_CURL_OPTS="${CURL_OPTS}" \
    && export PYTHON_BUILD_WGET_OPTS="${WGET_OPTS}" \
    && export PYTHON_CONFIGURE_OPTS=--disable-shared \
    && if [ ${PYTHON_VERSION_MINOR} -ge 10 ]; then export CPPFLAGS=-I/usr/include/openssl11 && export LDFLAGS=-L/usr/lib64/openssl11; fi \
    && eval "$(pyenv init -)" \
    && python-build `pyenv latest -k ${PYVERNAME}` /usr/local \
    && pyenv global system \
    && python3 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip \
    && rm -rf ~/.pyenv \
    && apt-get autoremove --purge -y \
       git \
       make \
       build-essential \
       libssl-dev \
       zlib1g-dev \
       libbz2-dev \
       libreadline-dev \
       libsqlite3-dev \
       wget \
       llvm \
       libncursesw5-dev \
       xz-utils \
       tk-dev \
       libxml2-dev \
       libxmlsec1-dev \
       libffi-dev \
       liblzma-dev \
    && pip3 install ${PIP_INS_OPTS} --no-cache-dir wheel protobuf \
    && pip3 install ${PIP_INS_OPTS} --no-cache-dir opencv-python || true \
    && ls /tmp/ \
    && rm -rf /tmp/*

ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH



#RUN apt update \
#    && apt install -y --no-install-recommends \
#       bzip2 \
#       ca-certificates \
#       curl \
#       apache2 \
#       libapache2-mod-wsgi-py3 \
#    && rm -rf /var/lib/apt/lists/*

#RUN umask 0 \
#    && mkdir -p /tmp/deps \
#    && cd /tmp/deps \
#    && curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh \
#    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
#    && rm -rf Miniconda3-latest-Linux-x86_64.sh \
#    && PATH=/opt/miniconda3/bin:$PATH \
#    && conda install python=3.7 \
#    && conda install pip wheel opencv \
#    && cd / \
#    && rm -rf /tmp/*
#ENV PATH /opt/miniconda3/bin:$PATH
EXPOSE 80

ADD . /code
WORKDIR /code
RUN pip3 install nnabla flask-cors
CMD python3.9 app.py
