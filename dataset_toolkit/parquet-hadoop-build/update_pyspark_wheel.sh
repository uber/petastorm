#!/bin/bash
set -eux -o pipefail

DIR=$(dirname $(readlink -e $0))
source "${DIR}/util.sh"
source "${DIR}/parquet-hadoop.version"

PYSPARK_VERSION="2.1.2"
# Revise the post-version number as needed when version updates are made.
PYSPARK_VERSION_SUFFIX=".post0+atg.pqhd"

DO_UPLOAD=0

function help_usage() {
    set +x
    echo "Usage: $0 [option] [pyspark_version]"
    echo ""
    echo "  -p <version>  Version string for parquet-hadoop.jar (default: ${PQHD_VERSION})"
    echo "  -u            Upload wheel to ATG artifactory (default: SKIP upload)"
    echo ""
    echo "Given a pyspark version, creates and uploads a PyPi wheel to ATG artifactory"
    echo "(${ARTIFACTORY_URL}), which"
    echo "contains an overridden parquet-hadoop-${PQHD_VERSION}.jar."
    echo ""
    echo "If no <pyspark_version> is supplied, this script defaults to ${PYSPARK_VERSION}."
    echo ""
    echo "Assumes the latest parquet-hadoop.jar exists in opus artifactory."
    echo ""
    echo "This should NOT need to be done frequently."
    echo ""
    echo "E.g.,"
    echo " $0 => Downloads source, overrides .jar, and builds wheel, saving it in ./"
    echo " $0 -u => Do the previous but upload wheel to ${ARTIFACTORY_URL}"
}

function download_pyspark_source() {
    local TMP=$1
    local VERSION=$2

    if [[ ! -x "$(command -v pip)" ]]; then
        error "You must have pip installed to proceed."
        exit -1
    fi

    local PYPI_PKG="pyspark==${VERSION}"
    trace "Downloading ${PYPI_PKG} into ${TMP}" >&2
    cd ${TMP}
    pip download ${PYPI_PKG} >&2
    local TGZ=$(find . -maxdepth 1 -type f -name "pyspark*${VERSION}*gz")
    tar zxf ${TGZ} >&2

    # Return full path to pyspark source directory
    find ${TMP} -maxdepth 1 -type d -name "pyspark-*" || exit -9
}

# This function assumes that check_artifactory_pypi_setup has been run.
function override_build_pyspark_wheel() {
    local PYSPARK=$1
    local PYSPARK_VERSION=$2
    local PYSPARK_PATCHED_VERSION=$3
    local JAR_PATH=$4
    local DO_UPLOAD=$5

    local PYSPARK_PQHD_JAR="${PYSPARK}/deps/jars/$(basename ${JAR_PATH})"
    local PYSPARK_VERSION_FILE="${PYSPARK}/pyspark/version.py"

    # Sanity check: If jar version doesn't match stock, something is wrong with pyspark version!
    if [[ ! -e ${PYSPARK_PQHD_JAR} ]]; then
        error "Failed to find version of JAR in pyspark source (${PYSPARK}): ${PYSPARK_PQHD_JAR}" \
              "Did you specify the appropriate pyspark <-> parquet-hadoop version combo?!"
        exit -9
    fi
    if [[ ! -e ${PYSPARK_VERSION_FILE} ]]; then
        error "Cannot find ${PYSPARK_VERSION_FILE}, which is totally unexpected!"
        exit -9
    fi

    # Simply: move .jar into place, overriding stock version; update __version__; then build wheel!
    mv ${JAR_PATH} ${PYSPARK_PQHD_JAR}
    pushd ${PYSPARK} &>/dev/null
    cat ${PYSPARK_VERSION_FILE} | sed -e 's=\(.*\)\('"${PYSPARK_VERSION}"'\)\(.*\)=\1'"${PYSPARK_PATCHED_VERSION}"'\3=' > ${PYSPARK_VERSION_FILE}~
    mv ${PYSPARK_VERSION_FILE}~ ${PYSPARK_VERSION_FILE}
    if [[ ${DO_UPLOAD} -eq 1 ]]; then
        trace "Building wheel and uploading to ATG artifactory: ${ARTIFACTORY_URL}" >&2
        python setup.py bdist_wheel upload -r local >&2
    else
        trace "Building wheel without artifactory upload" >&2
        python setup.py bdist_wheel >&2
    fi

    # Return full path to built pyspark wheel
    readlink -e "$(find dist -maxdepth 1 -type f -name "pyspark-${PYSPARK_PATCHED_VERSION}*.whl")"
}


while getopts ":hp:u" opt; do
    case $opt in
    h)
        help_usage
        exit -1
        ;;
    p)
        PQHD_VERSION=$OPTARG
        ;;
    u)
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

TMP_DIR=$(mktemp -d)
set +x

if [[ $# -ge 1 ]]; then
    PYSPARK_VERSION="$1"
else
    trace "No pyspark version specified. Using ${PYSPARK_VERSION}"
fi
PYSPARK_PATCHED_VERSION="${PYSPARK_VERSION}${PYSPARK_VERSION_SUFFIX}.${PQHD_VERSION}"

# Program dependency checks
check_opus_script
if [[ ${DO_UPLOAD} -eq 1 ]]; then
    check_artifactory_pypi_setup
fi

PYSPARK_PATH="$(download_pyspark_source ${TMP_DIR} ${PYSPARK_VERSION})"
trace "Source pyspark unpacked to ${PYSPARK_PATH}"
JAR_PATH="$(get_latest_parquet_hadoop_jar_from_opus ${PQHD_VERSION})"
trace "Latest parquet-hadoop JAR downloaded to ${JAR_PATH}"
WHEEL_PATH="$(override_build_pyspark_wheel ${PYSPARK_PATH} ${PYSPARK_VERSION} ${PYSPARK_PATCHED_VERSION} ${JAR_PATH} ${DO_UPLOAD})"

trace "========================================"
trace "Built custom pyspark ${PYSPARK_VERSION} wheel with parquet-hadoop.jar overridden."
trace "= JAR overridden: ${JAR_PATH}"
if [[ ${DO_UPLOAD} -eq 1 ]]; then
    trace "= pyspark wheel:  ${WHEEL_PATH}"
    trace "= Uploaded to ${ARTIFACTORY_URL}"
    trace ""
    REQ_FILE="$(readlink -e ${DIR}/../../../../../..)/requirements.txt"
    trace "Please update+commit pyspark version in ${REQ_FILE} to \"${PYSPARK_PATCHED_VERSION}\""
    grep -H -C1 pyspark ${REQ_FILE}
else
    mv ${WHEEL_PATH} .
    trace "= pyspark wheel:  $(basename ${WHEEL_PATH})"
fi
trace "========================================"

rm -rf ${TMP_DIR}
