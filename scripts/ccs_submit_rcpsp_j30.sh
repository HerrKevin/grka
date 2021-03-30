#!/bin/bash

#CCS --res="rset=1:ncpus=1:mem=4g:smp=f+3:ncpus=16:mem=60g"
#CCS -t 2d
#CCS --join oe
#CCS --output /upb/departments/pc2/users/k/kbt/research/grka/tuning/cluster_junk/tuninglog_rcpsp_j30.%reqid.out
#CCS -N gtune_rcpsp_j30

# NOTE: :smp=f if doing runtime tuning

DGGA_NODES=3 # NOTE update above before the second ncpus! Note: population is determined automatically in the run script
rnum=$(shuf -i 100-999 -n 1)
DGGA_PORT=10${rnum}
PYDGGA="/upb/departments/pc2/users/k/kbt/research/grka/pydgga"
TUNE_DIR="/scratch/hpc-prf-winf4/kbt/research/grka/"
SCENARIO="/scratch/hpc-prf-winf4/kbt/research/grka/tuning/scenario_rcpsp_j30/"
WORKER_DIR="/scratch/hpc-prf-winf4/kbt/research/grka/tuning/worker_output/"

OUT_FILE=${TUNE_DIR}/tuning/output/grka_rcpsp_j30_cpu120s_me25k_${CCS_REQID}.out.gz

cd ${TUNE_DIR}

ccsworker pshell -- ./scripts/ccs_run.sh "${PYDGGA}" "${DGGA_NODES}" "${DGGA_PORT}" "${OUT_FILE}" "${WORKER_DIR}" "${TUNE_DIR}" "${SCENARIO}"





