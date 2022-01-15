#!/bin/bash
MODELS=$1
T="../data"

python3 data_wrappers/tfrecord_wrapper.py $T --models $MODELS --sbp
