FROM ubuntu:20.04
# This container installs all of Petastorm's dependencies and can
# be used via the Makefile as an isolated workspace for Petastorm development.

# Install dependencies
RUN apt-get update -qq
RUN apt-get install -qq -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update -qq

RUN apt-get -qq install -y \
  build-essential \
  cmake \
  openjdk-8-jre-headless \
  git \
  python \
  python3-pip \
  python3.7 \
  python3.7-dev \
  python3.7-venv \
  python3.9 \
  python3.9-dev \
  python3.9-venv \
  virtualenv \
  wget \
  && rm -rf /var/lib/apt/lists/*


# Download and bake the mnist data into the docker image
RUN wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 -P /data/mnist/

# Temporarily add files needed for env setup.
RUN mkdir /petastorm
ADD setup.py /petastorm/
ADD README.rst /petastorm/
ADD petastorm /petastorm/petastorm

# Set up Python3 environment
RUN python3.7 -m pip install pip --upgrade
RUN python3.7 -m pip install wheel
RUN python3.7 -m venv /petastorm_venv3.7
RUN /petastorm_venv3.7/bin/pip3.7 install --no-cache scikit-build
RUN /petastorm_venv3.7/bin/pip3.7 install --no-cache -e /petastorm/[test,tf,torch,docs,opencv] --only-binary pyarrow --only-binary opencv-python
RUN /petastorm_venv3.7/bin/pip3.7 uninstall -y petastorm
# To avoid some version incompatibilities, we pin these libraries to versions that known to work together
RUN /petastorm_venv3.7/bin/pip3.7 install -U pyarrow==3.0.0 numpy==1.19.1 tensorflow==2.1.0 pyspark==3.0.0

# Otherwise we might have trouble with loading of libGL.so.1
RUN /petastorm_venv3.7/bin/pip3.7 install opencv-python-headless



# Set up Python3 environment
RUN python3.9 -m pip install pip --upgrade
RUN python3.9 -m pip install wheel
RUN python3.9 -m venv /petastorm_venv3.9
RUN /petastorm_venv3.9/bin/pip3.9 install --no-cache scikit-build
RUN /petastorm_venv3.9/bin/pip3.9 install --no-cache -e /petastorm/[test,tf,torch,docs,opencv] --only-binary pyarrow --only-binary opencv-python
RUN /petastorm_venv3.9/bin/pip3.9 uninstall -y petastorm
# To avoid some version incompatibilities, we pin these libraries to versions that known to work together
RUN /petastorm_venv3.9/bin/pip3.9 install -U pyarrow==3.0.0 numpy==1.19.3 tensorflow==2.5.0 pyspark==3.0.0

# Otherwise we might have trouble with loading of libGL.so.1
RUN /petastorm_venv3.9/bin/pip3.9 install opencv-python-headless


# Clean up
RUN rm -r /petastorm

ADD docker/run_in_venv.sh /
