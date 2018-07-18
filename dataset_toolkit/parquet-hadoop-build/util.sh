#!/bin/bash
set -eu -o pipefail

function error() {
    if [ -t 1 ]; then tput setaf 1; fi
    echo 'ERROR! '$* >&2
    if [ -t 1 ]; then tput sgr0; fi
}

function trace() {
    if [ -t 1 ]; then tput setaf 6; fi
    echo $*
    if [ -t 1 ]; then tput sgr0; fi
}

function check_opus_script() {
    if [[ ! -x "$(command -v opus)" ]] ; then
        set +x
        error 'Please follow the "Accessing Opus from a Workstation in the CORP Domain" section of' \
              'https://code.uberinternal.com/w/teams/map_creation/infrastructure_storage_and_compute/runbooks/opussetup/opus_cli/' \
              'to download and install the "opus.sh" script, and then retry the upload option.'
        exit 42
    fi
}

ARTIFACTORY_URL="https://artifacts.int.uberatc.com/artifactory/api/pypi/pypi-local"

function check_artifactory_pypi_setup() {
    if [[ ! -e "${HOME}/.pypirc" ]]; then
        set +x
        error 'Cannot find ~/.pypirc!'
        exit 42
    fi
    grep ${ARTIFACTORY_URL} ${HOME}/.pypirc
    if [[ $? -ne 0 ]]; then
        set +x
        error 'No pypi setup found for artifactory in ~/.pypirc! Please obtain Deploy snippets from' \
              'https://artifacts.int.uberatc.com/artifactory/webapp/#/artifacts/browse/tree/General/pypi-local'
        echo 'Consider excluding the password: field from .pypirc, as the upload step will prompt you.'
        exit 42
    fi
}

# This function assumes that check_opus_script has been run.
function get_latest_parquet_hadoop_jar_from_opus() {
    local JAR_VERSION=$1
    local JAR_NAME="parquet-hadoop-${JAR_VERSION}.jar"
    local COORDINATE="atg.parquet-hadoop:${JAR_NAME}"

    # Determine version of JAR to download from opus artifactory; list should show latest last
    TMP_FILE=$(mktemp)
    trace "Listing artifacts at ${COORDINATE}..." >&2
    opus artifact list ${COORDINATE} >${TMP_FILE}
    if [[ $? -ne 0 ]]; then
        error "Failed listing ${COORDINATE} in opus artifactory."
        exit -9
    fi
    local ARTIFACT_VERSION=$(tail -1 ${TMP_FILE})
    rm ${TMP_FILE}

    # OK, ready to download the JAR from opus artifact
    trace "Getting opus artifact [${ARTIFACT_VERSION}]..." >&2
    opus artifact get ${COORDINATE}:${ARTIFACT_VERSION} ${JAR_NAME} >&2
    if [[ $? -ne 0 ]]; then
        error "Failed to download ${JAR_NAME} from opus artifactory."
        exit -9
    fi

    # Return full path to JAR file
    readlink -e ${JAR_NAME}
}
