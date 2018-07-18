FROM docker.int.uberatc.com/services/shared/service_base:latest

MAINTAINER ATG-Data-Integration-Group <atg-data-int-group@uber.com>

# Nominal image name
LABEL com.uberatc.int.nominal-image-name='docker.int.uberatc.com/test/spark/parquet-hadoop'

# Get Oracle JDK 8 installed, in one layer, an no touch ever again. :-)
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

# Placing arguments here, as JDK install is unlikely to change, but
# the other package versions might.
#
ARG NUMPY_VERSION
ARG THRIFT_VERSION=0.7.0
ARG USER_GROUP=root:root

# Upgrade packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get -y upgrade

# Install various additional packages
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get -o Dpkg::Options::="--force-confmiss" -o Dpkg::Options::="--force-confold" \
    install -y \
      gawk \
      libboost-dev \
      libboost-filesystem-dev \
      libboost-system-dev \
      libssl-dev \
      libtool \
      automake \
      flex \
      bison \
      make \
      pkg-config \
      g++ \
      maven \
      && \
    apt-get clean && rm -rf /var/lib/apt/lists

# Build an old version of thrift!
RUN cd /tmp && \
    curl http://archive.apache.org/dist/thrift/${THRIFT_VERSION}/thrift-${THRIFT_VERSION}.tar.gz | tar zx && \
    cd thrift-${THRIFT_VERSION}/ && \
    bash configure && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/thrift-${THRIFT_VERSION}

# Obtain protobuf 3
RUN cd /tmp && \
    curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip && \
    unzip protoc-3.2.0-linux-x86_64.zip -d protoc3 && \
    sudo mv protoc3/bin/protoc /usr/bin/protoc && \
    sudo chmod a+rX /usr/bin/protoc && \
    cd / && \
    rm -rf /tmp/protoc3

RUN pip install --upgrade six numpy==${NUMPY_VERSION} pandas cython pytest cmake wheel

# Set default user and directory for container run to build package
RUN mkdir -p /atg-parquet-hadoop
USER ${USER_GROUP}
WORKDIR /atg-parquet-hadoop
