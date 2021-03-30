#!/usr/bin/python3

import sys
from sh import zgrep
import sh
import re
import os.path
import pandas as pd
import numpy as np
from scipy.stats import gmean
import traceback
from collections import defaultdict

bof = re.compile(r'^.*/\d+_(j\d+_\d+)\.sm\.out\.gz:.*grka:main:\d+ - Best objective found: (\d+\.\d*)$')
fname = re.compile(r'^.*/\d+_(j\d+_\d+)\.sm\.out\.gz$')

def analyze(inst_file, ddir):
    res = defaultdict(list)
    glb = sh.glob(f"{ddir}/*.gz")
    found = {}
    for ll in zgrep("Best objective found:", glb, _ok_code=[0,1]):
        mm = bof.match(ll)
        if mm:
            gg = mm.groups()
            inst = int(gg[0])
#             seed = int(gg[2])
            obj = float(gg[1])
            res['instance'].append(inst)
#             res['seed'].append(seed)
            res['objective'].append(obj)
            found[inst] = True
    with open(inst_file, 'r') as fp:
        lines = [line.strip() for line in fp]
    if len(lines) != len(res):
        for line in lines:
            inst = os.path.splitext(os.path.basename(line.split()[-1]))[0]
            if inst not in found:
                # Try to get the last objective function
                fail = True
                try:
                    last_line = list(zgrep("New best objective", sh.glob(f"{ddir}/*_{inst}.sm.out.gz"), _ok_code=[0,1,2]))[-1]
                    obj = float(last_line.split()[-1])
                    res['instance'].append(inst)
                    res['objective'].append(obj)
                    fail = False
                except:
                    pass
                    #traceback.print_exc()
                if fail:
                    print(f"Final output for {inst} is missing!")

    df = pd.DataFrame(res)
    return df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 analyze_rcpsp.py <inst_file> <directory to analyze>")
        sys.exit(1)

    inst_file = sys.argv[1]
    ddir = sys.argv[2]
    df = analyze(inst_file, ddir)
    print(f"Count {len(df)}; Mean (std): {df['objective'].mean():.3f} ({df['objective'].std():.3f}); Geo mean: {gmean(df['objective']):.3f}")
    up_dir = os.path.abspath(os.path.join(ddir, ".."))
    bname = os.path.basename(os.path.abspath(ddir))
    df.to_csv(f'{up_dir}/{bname}_agg.csv.gz')




