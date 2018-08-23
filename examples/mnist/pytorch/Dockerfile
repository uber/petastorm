#  Copyright (c) 2017-2018 Uber Technologies, Inc.
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
####
# Docker image built from pytorch for developing petastorm usage via pytorch.
####
FROM pytorch:latest

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:vbernat/haproxy-1.5 && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        nano \
        python \
        ca-certificates \
        libsm6 libxext6 libxrender-dev &&\
     rm -rf /var/lib/apt/lists/*

# Get Oracle JDK 8 installed, in one layer, and no touch ever again. :-)
RUN add-apt-repository ppa:webupd8team/java -y \
      && \
    echo debconf shared/accepted-oracle-license-v1-1 select true \
      | debconf-set-selections \
      && \
    echo debconf shared/accepted-oracle-license-v1-1 seen true \
      | debconf-set-selections \
      && \
    DEBIAN_FRONTEND=noninteractive apt-get update \
      && \
    apt-get -o Dpkg::Options::="--force-confmiss" -o Dpkg::Options::="--force-confold" \
    install -y \
      oracle-java8-installer \
      oracle-java8-set-default

RUN curl -o - https://bootstrap.pypa.io/get-pip.py | python - pip
RUN pip install --upgrade setuptools pip==18.0

WORKDIR /workspace
RUN chmod -R a+w /workspace

# Pull petastorm repo master
RUN git clone https://github.com/uber/petastorm.git

# OR, use this recipe to pull and build a particular pull request.
# Change to your fork repo path as appropriate.
#RUN git clone --depth=50 https://github.com/uber/petastorm.git && \
#    cd petastorm && \
#    git fetch origin +refs/pull/50/merge: && \
#    git checkout -qf FETCH_HEAD

RUN cd petastorm && \
    pip install -e .[opencv,tf,test] && \
    pip install --upgrade numpy
