#!/bin/bash

#CCS --res="rset=1:ncpus=1:mem=4g"
#CCS -t 6h
#CCS -J 0-16
#CCS --join oe
#CCS --output /upb/departments/pc2/users/k/kbt/research/grka/tuning/cluster_junk/bmark.%reqid.out
#CCS -N bmark_rcpsp

# NOTE: :smp=f if doing runtime tuning

module add lang/Python/3.8.2-GCCcore-9.3.0

splits=16 # should be b from -J a-b

source $1

out_dir=tuning/benchmark/rcpsp/j30/${CCS_ARRAY_ID}_${job_name}

grka_dir=/upb/departments/pc2/users/k/kbt/research/grka/
cd ${grka_dir}

source grkaenv/bin/activate

mkdir -p ${out_dir}


lines=$(wc -l ${inst_file} | cut -d " " -f1)
nom=$(( lines + ${splits} - 1))
split_size=$(( ${nom} / ${splits} ))
lstart=$((${CCS_ARRAY_INDEX} * ${split_size} + 1))
lend=$(( ${lstart} + ${split_size} - 1 ))

# echo $lstart $lend $split_size

if (( ${lstart} > ${lines} )); then
    exit 0
fi

IFS=$'\n'
for seed_inst in $(sed -n ${lstart},${lend}p ${inst_file}); do
    seed=$(echo "${seed_inst}" | cut -d " " -f1)
    inst=$(echo "${seed_inst}" | cut -d " " -f2 )
    binst=$(basename ${inst})
    out_file=${out_dir}/${CCS_ARRAY_ID}_${binst}.out.gz

    IFS=$' '
    echo "python3 grka.py rcpsp ${inst} --max_cpu 60 --max_evals ${max_evals} --seed ${seed} ${params} 2>&1 | gzip &> ${out_file}"
    python3 grka.py rcpsp ${inst} --max_cpu 60 --max_evals ${max_evals} --seed ${seed} ${params} 2>&1 | gzip &> ${out_file}
    IFS=$'\n'
done


