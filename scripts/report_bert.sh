#!/bin/bash
TASKS=$1
MODEL=$2
MAX_LAYER=$3
D="../experiments/${MODEL}-intercept"
T="../data"

for LAYER in $(seq 0 ${MAX_LAYER}); do
  # shellcheck disable=SC2068
	python3 report.py ${D} ${T}  --model ${MODEL}  --layer-index ${LAYER}  \
	--tasks $TASKS --clip-norm 1.0 --learning-rate 0.02 --batch-size 10 --ortho 0.1 #--gate-threshold 0.01
done
