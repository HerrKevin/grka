#!/usr/bin/python3

import sys
import sh
import os.path
from analyze_rcpsp import analyze
import re
import pandas as pd
import traceback

csv_re = re.compile(r'^\d+_(train|test)_(?:rcpsp_)?(j\d+)_(mod)?_?(.+)_(default|best)/?$')

def analyze_all(igrp, gstr):
    df_all = None
    for dd in sh.glob(gstr):
        print(dd)
        if os.path.isdir(dd):
            bname = os.path.basename(os.path.abspath(dd))
            up_dir = os.path.abspath(os.path.join(dd, ".."))
            csv = f'{up_dir}/{bname}_agg.csv.gz'
            if os.path.exists(csv):
                df = pd.read_csv(csv, index_col=0)
            else:
                try:
                    if 'train' in bname:
                        inst_file = f'../tuning/scenario_rcpsp_{igrp}/instances.txt'
                    else:
                        inst_file = f'../tuning/scenario_rcpsp_{igrp}/instances_test.txt'
                    df = analyze(inst_file, dd)
                except:
                    print(f"Failed to analyze {dd}")
                    traceback.print_exc()
                df.to_csv(csv)
            
            mm = csv_re.match(bname)
            if mm:
                gg = mm.groups()
                df['train_test'] = gg[0]
                df['inst_group'] = gg[1]
                df['solver'] = gg[-2]
                if df_all is None:
                    df_all = df
                else:
                    df_all = df_all.append(df, ignore_index=True)
            else:
                print(f"Failed to parse: {bname}")
    return df_all

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 analyze_all.py <j30/j60/...> <glob string for benchmarks/func>")
        sys.exit(1)
    igrp = sys.argv[1]
    gstr = sys.argv[2]
    df_all = analyze_all(igrp, f"../tuning/benchmark/rcpsp/{gstr}")
    print(len(df_all))
    df_all.reset_index()
    print(df_all)
    if df_all is not None:
        repstr = gstr.replace("/","") # don't use anything more complicated than this...
        repstr = repstr.replace("*","") # don't use anything more complicated than this...
        print(f"Output to: ../tuning/benchmark/rcpsp/aggregated/agg_{repstr}.csv.gz")
        df_all.to_csv(f"../tuning/benchmark/rcpsp/aggregated/agg_{repstr}.csv.gz")

