#!/bin/bash -e
#
# Uber, Inc. (c) 2018
#

if [ "$#" -ne 1 ]; then
    echo "Package dataset toolkit into a python wheel."
    echo
    echo "Syntax:"
    echo "  $0 <distribition directory>"
    echo
    echo "'distribition directory' will be created and a python wheel containing dataset_toolkit and"
    echo "its dependencies will be saved there"
    exit 1
fi


# This is where the wheel will be written to
DIST_DIR=$1

# The directory where this script is stored
THIS_DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# Root of git workspace
WORKSPACE_ROOT=$(git rev-parse --show-toplevel)

# Python interpreter used for packaging
PYTHON=${WORKSPACE_ROOT}/bin/python

setup_staging() {
    # The following av packages go into staging directory. We strip package prefix 
    # av.ml.dataset_toolkit -> dataset_toolkit
    # av.experimental.deepdrive.workers_pool    -> workers_pool
    # atcpy.hdfs

    STAGING_DIRECTORY=$1
    mkdir -p ${STAGING_DIRECTORY}/atcpy

    # Stage av.experimental.deepdrive.workers_pool
    cp -r ${WORKSPACE_ROOT}/source/python/av/experimental/deepdrive/workers_pool ${STAGING_DIRECTORY}

    # Stage av.ml.dataset_toolkit
    cp -r ${WORKSPACE_ROOT}/source/python/av/ml/dataset_toolkit ${STAGING_DIRECTORY}

    # TODO(yevgeni): integration_tests should not have been placed here from the start. Need to move it
    # out. For now, exclude it from the staging area
    rm -rf ${STAGING_DIRECTORY}/dataset_toolkit/integration_tests

    # For now we depend on the HDFS resolution code we implemented. Maybe would be able to use stock
    # libhdfs3 or other implementation when becomes available. For now need to packge this as well
    cp -r ${WORKSPACE_ROOT}/source/python/atcpy/hdfs ${STAGING_DIRECTORY}/atcpy
    touch ${STAGING_DIRECTORY}/atcpy/__init__.py    
}

edit_package_names() {
    # Edit package names - simple replace with sed:
    #  av.ml.dataset_toolkit -> dataset_toolkit
    #  av.experimental.deepdrive.workers_pool    -> workers_pool
    STAGING_DIRECTORY=$1
    cd ${STAGING_DIRECTORY}

    # Explicitly specifying directory argument to 'find' to make editing a little safer 
    find dataset_toolkit -name "*py" | xargs sed -i "s/av\.ml\.dataset_toolkit/dataset_toolkit/g"
    find dataset_toolkit -name "*py" | xargs sed -i "s/av\.experimental\.deepdrive\.workers_pool/workers_pool/g"
    find workers_pool -name "*py" | xargs sed -i "s/av\.experimental\.deepdrive\.workers_pool/workers_pool/g"
}

test_package() {
    # Install newly created package into a clean virtualenv
    # Run unit tests that we packaged together with the code (is it a good idea?)

    DIST_DIR=$1
    cd $(mktemp -d)
    virtualenv --no-site-packages .env
    source .env/bin/activate
    pip install ${DIST_DIR}/*whl
    python -m unittest discover .env/lib/python2.7/site-packages/dataset_toolkit 
    python -m unittest discover .env/lib/python2.7/site-packages/workers_pool/ 
    python -m unittest discover .env/lib/python2.7/site-packages/atcpy/
    deactivate
}

STAGING_DIRECTORY=$(mktemp -d)
trap "rm -rf ${STAGING_DIRECTORY}" EXIT

echo -e "\n\n********************* Setting up staging directory *********************"
setup_staging ${STAGING_DIRECTORY}
edit_package_names ${STAGING_DIRECTORY}

echo -e "\n\n********************* Creating Python Wheel ********************* "
# Generate the package
${PYTHON} ${THIS_DIR}/setup.py sdist bdist_wheel --dist-dir ${DIST_DIR}

echo -e "\n\n********************* Testing package ${DIST_DIR}/*whl *********************"
test_package ${DIST_DIR}

echo Done
