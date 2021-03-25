#!/bin/bash

#CCS --res="rset=1:ncpus=1:mem=4g"
#CCS -t 4h
#CCS -J 0-16
#CCS --join oe
#CCS --output /upb/departments/pc2/users/k/kbt/research/grka/tuning/cluster_junk/bmark.%reqid.out
#CCS -N bmark_Fd20

# NOTE: :smp=f if doing runtime tuning

module add lang/Python/3.8.2-GCCcore-9.3.0

splits=16 # should be b from -J a-b

source $1

out_dir=tuning/benchmark/func/${CCS_ARRAY_ID}_${job_name}

grka_dir=/upb/departments/pc2/users/k/kbt/research/grka/
cd ${grka_dir}

source grkaenv/bin/activate

mkdir -p ${out_dir}

lines=$(wc -l ${inst_file} | cut -d " " -f1)
nom=$(( lines + ${splits} - 1))
split_size=$(( ${nom} / ${splits} ))
lstart=$((${CCS_ARRAY_INDEX} * ${split_size} + 1))
lend=$(( ${lstart} + ${split_size} - 1 ))

echo $lstart $lend $split_size

if (( ${lstart} > ${lines} )); then
    exit 0
fi

for inst in $(sed -n ${lstart},${lend}p ${inst_file} | cut -d " " -f2); do
    out_file=${out_dir}/${CCS_ARRAY_ID}_${inst}.out.gz

    IFS='_' read -ra sp <<< "${inst}"

    func=${sp[0]}
    finst=${sp[1]}
    seed=${sp[2]}

    echo "python3 grka.py func /dev/null --use_gap --dims ${dims} --max_cpu 60 --max_evals ${max_evals} --func ${func} --finst ${finst} --seed ${seed} ${params} 2>&1 | gzip &> ${out_file}"
    python3 grka.py func /dev/null --use_gap --dims ${dims} --max_cpu 60 --max_evals ${max_evals} --func ${func} --finst ${finst} --seed ${seed} ${params} 2>&1 | gzip &> ${out_file}
done


