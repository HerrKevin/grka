#!/bin/bash

if [ -z $1 ];
    echo "Usage: ./bmark_submit_func.sh <config file>"
    exit 1
fi

config=${1}

grka_dir=/upb/departments/pc2/users/k/kbt/research/grka/
cd ${grka_dir}

ccsalloc scripts/_bmark_submit_func.sh ${config}

