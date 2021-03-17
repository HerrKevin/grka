#!/bin/bash

module add lang/Python/3.8.2-GCCcore-9.3.0

echo CL: $@

PYDGGA=${1}
DGGA_NODES=${2}
DGGA_PORT=${3}
OUT_FILE=${4}
WORKER_OUT=${5}
TUNE_DIR=${6}
SCENARIO=${7}
#shift 15
#FLAGS=$@

cd ${TUNE_DIR}

#POPULATION=$((${DGGA_NODES}*${MINT_SIZE}*2))

# 
# echo "Master running on ${master_ip}"

WORKER_OUT_DIR=${WORKER_OUT}/${CCS_REQID}/
mkdir -p ${WORKER_OUT_DIR}

master=$(python3 scripts/ccs_am_i_master.py ${CCS_MAPPING})
echo "Am I master? ${master}; me: $(hostname); CCS_MAPPING: ${CCS_MAPPING}"
if [ ${master} -eq 1 ]; then
    ## Execute dgga as master

    master_ip=$(ifconfig 2>&1 | grep 10.149 | cut -d " " -f10)
    echo IP: ${master_ip}
    echo "${PYDGGA}" -s ${SCENARIO} --slots 16 --num-workers ${DGGA_NODES} --port ${DGGA_PORT}

    "${PYDGGA}" dgga -s ${SCENARIO} --slots 16 --num-workers "${DGGA_NODES}" --worker-script scripts/fake_worker.sh --port "${DGGA_PORT}" --verbose DEBUG &> ${OUT_FILE}

else
    sleep 10 # wait 10 seconds before starting the workers so the master has time to start
    master_ip=$(python3 scripts/ccs_get_master_ip.py ${CCS_MAPPING})

    "${PYDGGA}" dggaw --address "${master_ip}" --slots 16 --port "${DGGA_PORT}" --verbose DEBUG &> ${WORKER_OUT_DIR}/$(basename ${OUT_FILE})_worker_$(hostname)_${CCS_REQID}.out
fi

