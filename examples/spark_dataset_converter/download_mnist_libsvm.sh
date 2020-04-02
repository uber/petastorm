#!/usr/bin/env bash

mkdir -p /tmp/petastorm/mnist
cd /tmp/petastorm/mnist || exit
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 -O mnist.bz2
