#!/bin/bash

#CCS --res="rset=1:ncpus=1:mem=4g:smp=f+4:ncpus=16:mem=60g:smp=f"
#CCS -t 4d
#CCS --join oe
#CCS --output /upb/departments/pc2/users/k/kbt/research/bbochallenge/cluster_junk/tuninglog.%reqid.out
#CCS -N l2gpydgga

DGGA_NODES=4 # NOTE update above before the second ncpus! Note: population is determined automatically in the run script
rnum=$(shuf -i 100-999 -n 1)
DGGA_PORT=10${rnum}
PYDGGA="/upb/departments/pc2/users/k/kbt/research/grka/pydgga"
TUNE_DIR="/scratch/hpc-prf-winf4/kbt/research/grka/"
SCENARIO="/scratch/hpc-prf-winf4/kbt/research/grka/tuning/scenario_bbo/"
WORKER_DIR="/scratch/hpc-prf-winf4/kbt/research/grka/tuning/worker_output/"

OUT_FILE=${TUNE_DIR}/output/bbo_l2_${CCS_REQID}.out

cd ${TUNE_DIR}

ccsworker pshell -- ./scripts/ccs_run.sh "${PYDGGA}" "${DGGA_NODES}" "${DGGA_PORT}" "${OUT_FILE}" "${WORKER_DIR}" "${TUNE_DIR}" "${SCENARIO}"





