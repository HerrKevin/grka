#!/bin/bash

module add lang/Python/3.8.2-GCCcore-9.3.0

if [ -z "$1" ]; then
    echo "Usage: ./run_gga_local.sh <scenario>"
    exit 1
fi

scenario=$1

./pydgga gga -s ${scenario}

