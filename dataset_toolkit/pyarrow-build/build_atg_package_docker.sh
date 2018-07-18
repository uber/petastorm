#!/usr/bin/env bash

set -e

function main() {
  if [ $# -lt 2 ] ; then
    echo "Missing command line parameter(s)"
    echo "Usage: ./build_atg_package_docker.sh <destination dir> <arrow tag/commit hash from > [--upload]"
    echo "Note that the tag/hash comes from the repo https://code.int.uberatc.com/source/apache-arrow/"
    exit 1
  fi

  DEST=$1
  if [[ ${DEST} == */ ]]; then
    DEST=${DEST: :-1}
  fi

  ARROW_VERSION=$2
  cd $(git rev-parse --show-toplevel)/source/ml/python/dataset_toolkit/pyarrow-build

  # We want to build against the version of numpy used in rAV to guarantee compatibility
  NUMPY_VERSION=$(grep "numpy" $(git rev-parse --show-toplevel)/requirements.lock | cut -f 3 -d '=')

  echo "Build Docker image for $(pwd) repo ..."
  docker build -f Dockerfile --build-arg NUMPY_VERSION=$NUMPY_VERSION -t pyarrow-atg .

  echo "Execute docker build..."
  if [[ ${DEST:0:1} == "/" ]]; then
    HOST_DIST=${DEST}
  else
    HOST_DIST=$(pwd)/${DEST}
  fi

  # If we don't create host dir here it will end up created with root as owner and we won't be able
  # to tar everything in later in the script
  mkdir -p ${HOST_DIST}

  # clone the arrow fork and checkout the requested version/commit
  git clone ssh://code@code.int.uberatc.com/source/apache-arrow.git $HOST_DIST/arrow
  pushd $HOST_DIST/arrow
  git checkout $ARROW_VERSION
  popd

  # clone parquet-cpp and checkout a recent commit that we know is compatible with recent versions of pyarrow
  git clone https://github.com/apache/parquet-cpp.git $HOST_DIST/parquet-cpp
  pushd $HOST_DIST/parquet-cpp
  # Tie to specific commit of parquet-cpp which we know works https://github.com/apache/parquet-cpp.git
  git checkout b62c22e52d37fa9580d85c5a2a2d73bd8e268364
  popd

  docker run --rm -v ${HOST_DIST}:/pyarrow-build-atg pyarrow-atg \
    ./build_atg_package.sh /pyarrow-build-atg

  if [[ "${3}" == "--upload" ]]; then
    echo "Uploading artifacts to artifactory"
    WHEEL_DIR=${HOST_DIST}/arrow/python/dist
    WHEEL_FILE=$(basename $(ls ${WHEEL_DIR}/*.whl))
    FULL_VERSION=$(echo $WHEEL_FILE | cut -d "-" -f 2)
    VERSION=${FULL_VERSION//.dev*/}
    curl -T ${WHEEL_DIR}/${WHEEL_FILE} \
      -H "X-Checksum-MD5:$(md5sum ${WHEEL_DIR}/${WHEEL_FILE} | awk '{print $1}')" \
      -H "X-Checksum-Sha1:$(shasum -a 1 ${WHEEL_DIR}/${WHEEL_FILE} | awk '{ print $1 }')" \
      -u ${USER} \
      -X PUT \
      https://docker.int.uberatc.com/artifactory/pypi-local/atg-pyarrow/${VERSION}/${WHEEL_FILE}
  fi

  echo "Done"
}

main "$@"
