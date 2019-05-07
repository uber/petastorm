FROM ubuntu:16.04
# This container installs all of Petastorm's dependencies and can
# be used via the Makefile as an isolated workspace for Petastorm development.

# Install dependencies
RUN apt-get update -qq
RUN apt-get install -qq -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update -qq
RUN apt-get -qq install -y \
  build-essential \
  default-jre \
  git \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  python-pip \
  python-software-properties \
  python2.7 \
  python2.7-dev \
  python3-pip \
  python3.6 \
  python3.6-dev \
  python3.6-venv \
  virtualenv \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Temporarily add files needed for env setup.
RUN mkdir /petastorm
ADD setup.py /petastorm/
ADD README.rst /petastorm/
ADD petastorm /petastorm/petastorm

# Set up Python3 environment
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel
RUN python3.6 -m venv /petastorm_venv3.6
RUN /petastorm_venv3.6/bin/pip3.6 install -e /petastorm/[test,tf,torch,docs,opencv]
RUN /petastorm_venv3.6/bin/pip3.6 uninstall -y petastorm

# Set up Python2 environment
RUN virtualenv /petastorm_venv2.7
RUN python2.7 -m pip install pip --upgrade
RUN /petastorm_venv2.7/bin/pip2.7 install -e /petastorm/[test,tf,torch,docs,opencv]
RUN /petastorm_venv2.7/bin/pip2.7 uninstall -y petastorm

# Install Hadoop, set environment
RUN wget http://archive.apache.org/dist/hadoop/common/hadoop-2.8.1/hadoop-2.8.1.tar.gz -O /root/hadoop-2.8.1.tar.gz
RUN tar xzf /root/hadoop-2.8.1.tar.gz --directory /root
RUN echo "export HADOOP_HOME=~/hadoop-2.8.1" >> ~/.bashrc
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/" >> ~/.bashrc
RUN echo "PATH=$PATH:~/hadoop-2.8.1/bin" >> ~/.bashrc

# Clean up
RUN rm -r /petastorm
