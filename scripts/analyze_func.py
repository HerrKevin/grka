#!/usr/bin/python3

import sys
from sh import zgrep
import sh
import re
import os.path
import pandas as pd
import numpy as np
from scipy.stats import gmean
from collections import defaultdict

bof = re.compile(r'^.*/\d+_(\d+)_(\d+)_(\d+)\.out\.gz:.*grka:main:\d+ - Best objective found: (\d+\.\d*)$')

def analyze(ddir):
    res = defaultdict(list)
    for ll in zgrep("Best objective found:", sh.glob(f"{ddir}/*.gz")):
        mm = bof.match(ll)
        if mm:
            gg = mm.groups()
            func = int(gg[0])
            inst = int(gg[1])
            seed = int(gg[2])
            obj = float(gg[3])
            res['function'].append(func)
            res['instance'].append(inst)
            res['seed'].append(seed)
            res['objective'].append(obj)

    df = pd.DataFrame(res)
    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <directory to analyze>")
        sys.exit(1)

    ddir = sys.argv[1]
    df = analyze(ddir)
    print(f"Count {len(df)}; Mean (std): {df['objective'].mean():.3f} ({df['objective'].std():.3f}); Geo mean: {gmean(df['objective']):.3f}")
    up_dir = os.path.abspath(os.path.join(ddir, ".."))
    bname = os.path.basename(os.path.abspath(ddir))
    df.to_csv(f'{up_dir}/{bname}_agg.csv.gz')




