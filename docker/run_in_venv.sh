#!/usr/bin/env bash

PYTHON_VER=$1

source /petastorm_venv${PYTHON_VER}/bin/activate
cd /petastorm

shift 1

"$@"
