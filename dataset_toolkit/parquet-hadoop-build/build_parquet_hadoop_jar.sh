#!/bin/bash
set -eux -o pipefail

DIR=$(dirname $(readlink -e $0))
source "${DIR}/util.sh"
source "${DIR}/parquet-hadoop.version"

PQRT_BRANCH="apache-parquet-${PQHD_VERSION}-atg"

SKIP_BUILD=0
DO_UPLOAD=0
DIR_TO_COPY_JAR=$(pwd)

# Build just parquet-hadoop and dependencies
MVN_PROJ_LIST="--also-make --projects parquet-hadoop"

function help_usage() {
    set +x
    echo "Usage: $0 [option] [prqt_path]"
    echo ""
    echo "  -a  Build the entire parquet-mr project (default: parquet-hadoop only)"
    echo "  -d  Directory to copy the built JAR when no <prqt_path> given (default: .)"
    echo "  -s  Skip new build of parquet-mr altogether (default: DO build)"
    echo "  -u  Upload JAR to opus artifactory (default: SKIP upload)"
    echo ""
    echo "Invokes a docker container to build a parquet-hadoop JAR."
    echo "The rPRQT branch built will be [${PQRT_BRANCH}]."
    echo "If no <prqt_path> is provided, one will be git-cloned to /tmp."
    echo "* Supply -u to upload JAR to opus artifactory."
    echo "* Option -d only takes effect if no <prqt_path> is supplied."
    echo ""
    echo "E.g.,"
    echo " $0 => Build from git and copy JAR to current directory"
    echo " $0 -u => Build from git and upload to opus"
    echo " $0 -s -u => Skip new build and updload built JAR in parquet-hadoop/target"
}

function checkout_atg_parquet_mr() {
    TMP=$(mktemp -d)
    echo "Checking out 'parquet-mr-atg' into ${TMP}"
    export PRQT_PATH=${TMP}/parquet-mr-atg
    git clone -b ${PQRT_BRANCH} --single-branch \
        ssh://code@code.int.uberatc.com/source/parquet-mr-atg.git \
        ${PRQT_PATH}
}

function do_dockered_package_build() {
    pushd ${DIR}

    set -x
    # We want to build against the version of numpy used in rAV to guarantee compatibility
    NUMPY_VERSION=$(grep "numpy" $(git rev-parse --show-toplevel)/requirements.lock | cut -f 3 -d '=')
    # We fetch the version required by parquet-mr, specified in the pom.xml
    THRIFT_VERSION=$(grep -A 20 thrift.executable ${PRQT_PATH}/pom.xml | grep thrift.version | sed -e 's=^.*>\(.*\)<.*$=\1=')

    # Docker build failure will fail the script; we're not trapping errors, yet.
    # So, just issue a prelim warning, but don't actually do docker restart (one fewer pwd prompt!)
    trace "If docker build fails for mysterious reasons, try 'sudo service docker restart'."
    docker build \
      --build-arg NUMPY_VERSION=${NUMPY_VERSION} \
      --build-arg THRIFT_VERSION=${THRIFT_VERSION} \
      --build-arg USER_GROUP="$(id -u):${GROUPS[0]}" \
       -t atg-parquet-hadoop .

    docker run --rm --user=$(id -u):${GROUPS[0]} -v ${PRQT_PATH}:/atg-parquet-hadoop atg-parquet-hadoop mvn -Drat.skip=true -DskipTests=true package ${MVN_PROJ_LIST}
    rv=$?
    set +x

    if [[ $rv -ne 0 ]]; then
        echo "Package build in docker failed! Please check output to determine issues."
        exit -9
    fi

    popd
}

# This function assumes that check_opus_script has been run.
function upload_built_jar_to_opus() {
    TARGET=$1
    VER_STR=$2
    VER_TAG=$3

    if [[ -e "${TARGET}" ]]; then
        echo "Uploading parquet-hadoop-${VER_STR}.jar to opus with version tag \"${VER_TAG}\"..."
        opus artifact put atg.parquet-hadoop:parquet-hadoop-${VER_STR}.jar:${VER_TAG} "${TARGET}"
        echo "Listing opus artifacts named \"atg.parquet-hadoop:parquet-hadoop-${VER_STR}.jar\":"
        opus artifact list atg.parquet-hadoop:parquet-hadoop-${VER_STR}.jar
    fi
}

while getopts ":had:su" opt; do
    case $opt in
    h)
        help_usage
        exit -1
        ;;
    d)
        DIR_TO_COPY_JAR=$(readlink -e $OPTARG)
        if [[ ! -d "${DIR_TO_COPY_JAR}" ]]; then
            echo "ERROR: Invalid directory \"${DIR_TO_COPY_JAR}\""
            echo "Please supply a valid directory."
            echo
            help_usage
            exit -2
        fi
        ;;
    a)
        # Do full build...in case that's ever needed for experimentation
        MVN_PROJ_LIST=""
        ;;
    s)
        SKIP_BUILD=1
        ;;
    u)
        check_opus_script
        DO_UPLOAD=1
        ;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        help_usage
        exit -2
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        help_usage
        exit -3
        ;;
    esac
done
shift $((OPTIND-1))

need_to_cleanup=""

if [[ $# -ge 1 ]]; then
    PRQT_PATH=$(readlink -e $1)
    # Disable copy of JAR
    DIR_TO_COPY_JAR=""
else
    # Checkout parquet-hadoop from git into temp dir
    checkout_atg_parquet_mr
    need_to_cleanup=$(readlink -e ${PRQT_PATH}/..)
fi

VER_STR=$(grep -A10 '<parent>' ${PRQT_PATH}/parquet-hadoop/pom.xml | grep '<version>' | sed -e 's=^.*>\(.*\)<.*$=\1=')
VER_TAG="$(date +"%Y%m%d.%H%M%S").${USER}"
JAR_NAME="parquet-hadoop-${VER_STR}.jar"
TARGET="${PRQT_PATH}/parquet-hadoop/target/${JAR_NAME}"
set +x

if [[ ${SKIP_BUILD} -ne 1 ]]; then
    do_dockered_package_build
fi

if [[ -e "${TARGET}" ]]; then
    final_target="${TARGET}"

    if [[ ${DO_UPLOAD} -eq 1 ]]; then
        upload_built_jar_to_opus "${TARGET}" "${VER_STR}" "${VER_TAG}"
    fi
    if [[ -d "${DIR_TO_COPY_JAR}" ]]; then
        # We need to copy JAR to new location and display that
        cp -p "${TARGET}" "${DIR_TO_COPY_JAR}/"
        final_target="${DIR_TO_COPY_JAR}/${JAR_NAME}"
    fi

    echo "========================================"
    echo "JAR location: ${final_target}"
    echo "========================================"
else
    echo "ERROR! Nonexistent build target ${TARGET}"
    echo 'You may have accidentally skipped build?!'
fi

if [[ -n "${need_to_cleanup}" ]]; then
    echo "Cleaning up ${need_to_cleanup}..."
    rm -rf ${need_to_cleanup}
fi
