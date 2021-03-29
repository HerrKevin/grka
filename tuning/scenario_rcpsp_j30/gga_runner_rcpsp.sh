#!/bin/bash

cd /scratch/hpc-prf-winf4/kbt/research/grka/
source grkaenv/bin/activate

inst=$1
seed=$3


python3 grka.py rcpsp ${inst} --seed ${seed} --max_cpu 60 --tuner --max_evals 10000 "${@:4}"

