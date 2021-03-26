#!/bin/bash

cd /scratch/hpc-prf-winf4/kbt/research/grka/
source grkaenv/bin/activate

inst=$1
seed=$3


python3 grka.py rcpsp ${inst} --dims 20 --seed ${seed} --max_cpu 60 --tuner --use_gap --max_evals 10000 "${@:4}"

