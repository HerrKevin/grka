#!/bin/bash

if [ -z $1 ]; then
    echo "Usage: ./bmark_submit_func.sh <config file>"
    exit 1
fi

config=${1}
abs_path=$(realpath ${config})

if [ ! -f ${abs_path} ]; then
    echo "Usage: ./bmark_submit_func.sh <config file>"
    exit 1
fi

grka_dir=/upb/departments/pc2/users/k/kbt/research/grka

ccsalloc ${grka_dir}/scripts/_bmark_submit_func_job.sh ${abs_path}

