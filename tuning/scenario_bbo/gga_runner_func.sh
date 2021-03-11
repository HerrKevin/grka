#!/bin/bash

cd /Users/kbt/Research/grka/

inst=$1

IFS='_' read -ra sp <<< "${inst}"

func=${sp[0]}
finst=${sp[1]}
seed=${sp[2]}

python3 grka.py func /dev/null --func ${func} --finst ${finst} --seed ${seed} --tuner --use_gap --max_evals 10000 "${@:4}"

