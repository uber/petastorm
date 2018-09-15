FROM ubuntu:16.04
RUN apt-get -qq update

RUN apt-get install -qq -y software-properties-common python-software-properties

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update -qq

RUN apt-get -qq install -y \
                build-essential \
                python3.6 \
                python3.6-dev \
                python3-pip \
                python3.6-venv \
                virtualenv \
                git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

RUN apt install -qq -y \
            git \
            libglib2.0-0 \
            libsm6 \
            libxrender1 \
            default-jre


RUN mkdir /petastorm

ADD README.rst /petastorm/
ADD setup.py /petastorm/
ADD petastorm /petastorm/petastorm

RUN python3.6 -m venv /petastorm_venv3.6
RUN /petastorm_venv3.6/bin/pip3.6 install -e /petastorm/[test,tf,torch,docs,opencv]
RUN /petastorm_venv3.6/bin/pip3.6 uninstall -y petastorm

RUN virtualenv /petastorm_venv2.7

RUN apt-get install -y \
                python2.7 \
                python2.7-dev

#RUN python2.7 -m pip install pip --upgrade

RUN /petastorm_venv2.7/bin/pip2.7 install -e /petastorm/[test,tf,torch,docs,opencv]
RUN /petastorm_venv2.7/bin/pip2.7 uninstall -y petastorm

ADD docker/run_in_venv.sh /
