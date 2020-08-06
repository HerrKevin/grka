#!/bin/bash

inst_dir=/Users/kbt/Research/dualcycleroro/data/ds_29_07_20
out_dir=output/roro

for alg in pso de brkga cmaes; do
    out_alg=${out_dir}/${alg}
    mkdir -p ${out_alg}
    for tt in 2 4; do
        for ii in ${inst_dir}/*.xlsx; do
            bname=$(basename ${ii})
            outf=${out_alg}/${bname%.*}_tugs${tt}.out.gz
            echo "python3 grka.py roro brkga ${ii} --max_cpu 60 --pop_size 500 --tugs ${tt} 2>&1 | gzip &> ${outf}"
        done
    done
done


