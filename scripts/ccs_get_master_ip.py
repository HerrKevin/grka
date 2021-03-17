#!/usr/bin/python3

import sys
from sh import hostname, nslookup, grep, tail

# test = "node01-009:=1:ompthreads=16:ncpus=16:mem=66060288k:vmem=66060288k,node01-008:=1:ncpus=1:mem=4194304k:vmem=4194304k"

ccs_mapping = sys.argv[1]
hname = hostname()

for rset in ccs_mapping.split(','):
    kvs = rset.split(":")
    node = kvs[0]
    for kv in kvs[1:]:
        key,val = kv.split("=")
        if key == 'ncpus':
            if int(val) == 1:
                raw = tail(grep(nslookup(node), "Address:"), "-1")
                ip = raw.split()[-1]
                print(ip)
                sys.exit(1)
print("0.0.0.0")
