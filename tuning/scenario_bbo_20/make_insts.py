#!/usr/bin/python3

import cma.bbobbenchmarks as bn
import random

train = [3,7,15,16,20,21]

for ff in range(1,25):
    if ff not in train:
        for ii in range(1, 21):
            _, opt = bn.instantiate(ff, ii)
            if opt > 0:
                for ss in range(10):
                    print(f"{ff}_{ii}_{random.randint(1,99999)}")

