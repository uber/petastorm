#!/usr/bin/env bash

set -e

function main() {
  if [ $# -lt 1 ] ; then
    echo "Missing command line parameter"
    echo "Usage: ./build-atg/build_atg_package.sh <destination dir>"
    exit 1
  fi

  DEST=$1
  # Remove a trailing /
  if [[ ${DEST} == */ ]]; then
    DEST=${DEST: :-1}
  fi

  # convert relative path to absolute
  if [[ ${DEST} != /* ]]; then
    echo "Script need absolute path to destination, convert '${DEST}'"
    DEST=$(pwd)/${DEST}
    echo "    to '${DEST}'"
  fi

  mkdir $DEST/dist

  export ARROW_BUILD_TYPE=release
  export ARROW_HOME=$DEST/dist
  export PARQUET_HOME=$DEST/dist
  export LD_LIBRARY_PATH=$DEST/dist/lib:$LD_LIBRARY_PATH

  mkdir $DEST/arrow/cpp/build
  pushd $DEST/arrow/cpp/build

  cmake -DCMAKE_BUILD_TYPE=$ARROW_BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX=$ARROW_HOME \
        -DARROW_PYTHON=on \
        -DARROW_PLASMA=on \
        -DARROW_BUILD_TESTS=OFF \
        ..
  make -j4
  make install
  popd

  mkdir $DEST/parquet-cpp/build
  pushd $DEST/parquet-cpp/build

  cmake -DCMAKE_BUILD_TYPE=$ARROW_BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX=$PARQUET_HOME \
        -DPARQUET_BUILD_BENCHMARKS=off \
        -DPARQUET_BUILD_EXECUTABLES=off \
        -DPARQUET_BUILD_TESTS=off \
        ..

  make -j4
  make install
  popd

  echo "Building pyarrow..."
  cd $DEST/arrow/python
  python setup.py build_ext --build-type=$ARROW_BUILD_TYPE \
       --with-parquet --with-plasma --inplace

 echo "Building pyarrow wheel..."
 python setup.py build_ext --build-type=$ARROW_BUILD_TYPE \
      --with-parquet --with-plasma --bundle-arrow-cpp bdist_wheel
}

main "$@"
