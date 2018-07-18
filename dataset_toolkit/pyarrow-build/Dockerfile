FROM docker-registry.pit-irn-1.uberatc.net/uber-uai/atg-python-flask-av:sjc1-produ-0000000005
ARG NUMPY_VERSION

RUN apt-get update && apt-get install -y libboost-dev \
    libboost-filesystem-dev \
    libboost-system-dev\
    flex bison

RUN pip install --upgrade six numpy==$NUMPY_VERSION pandas cython pytest cmake wheel

# Copy sources
RUN mkdir -p /pyarrow-atg
ADD . /pyarrow-atg/

# make output directory
RUN mkdir -p /pyarrow-build-atg

# Run build
WORKDIR /pyarrow-atg
