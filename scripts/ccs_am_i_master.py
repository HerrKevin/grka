#!/usr/bin/python3

import sys
from sh import hostname

# test = "node01-009:=1:ompthreads=16:ncpus=16:mem=66060288k:vmem=66060288k,node01-008:=1:ncpus=1:mem=4194304k:vmem=4194304k"
# test = "washington005:=1:ompthreads=16:ncpus=16:mem=66060288k:vmem=66060288k,node01-013:=1:ncpus=1:mem=4194304k:vmem=4194304k"

ccs_mapping = sys.argv[1]
hname = hostname()
hname = hname.strip()

for rset in ccs_mapping.split(','):
    kvs = rset.split(":")
    node = kvs[0]
    for kv in kvs[1:]:
        key,val = kv.split("=")
        if key == 'ncpus':
            if int(val) == 1 and hname == node:
                print(f"1")
                sys.exit(1)
print("0")


