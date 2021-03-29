#!/usr/bin/python3

import sys
import sh
from sh import zgrep
import re
import numpy as np
from loguru import logger
sys.path.append("../")
from problem_rcpsp import read_instance


gre = re.compile(r'^(.*(j\d+)_.*\d+_(j\d+_\d+).sm.out.gz):.*Job \(start, end\):\s+(\[.*\])$')

def check_feas(odir):
    for line in zgrep("problem_rcpsp:evaluate_fb", sh.glob(f"{odir}/*.out.gz"), _ok_code=[0,1,2]):
        if 'j309_4' not in line:
            continue
        mm = gre.match(line)
        if mm:
            gg = mm.groups()
            sol_file = gg[0]
            group = gg[1]
            inst_name = gg[2]
            sol = eval(gg[3])
#             if sol[0][0] > 1:
#                 sol.reverse()
            inst_path = f'../data/rcpsp/{group}/{inst_name}.sm'
            inst = read_instance(inst_path)

            # Check resource capacity constraint
            rusage = np.zeros((inst.resources, inst.horizon))
            for jj, (start, end) in enumerate(sol):
                if start == end:
                    continue
                rusage[:,start:end+1] += inst.usage[jj][:, np.newaxis]
            if (rusage > inst.rcap[:, np.newaxis]).any():
                print(rusage)
                print(inst.rcap)
                print(f"Instance {inst_name} infeasible (resource capacity exceeded). File: {sol_file}")
                continue

            # Check precedence relations
            jopen = [0]
            invalid = False
            while jopen and not invalid:
                jj = jopen[-1]
                del jopen[-1]
                jopen.extend(inst.succ[jj])
                for ss in inst.succ[jj]:
                    if sol[jj][0] >= sol[ss][1]:
                        print(f"Instance {inst_name} infeasible (precedence violated {jj}@{sol[jj]}->{ss}@{sol[ss]}). File: {sol_file}")
                        print(sol)
                        invalid = True
            if invalid:
                continue



if __name__ == "__main__":
    logger.remove()
    if len(sys.argv) != 2:
        print("Usage: python3 feas_check_rcpsp.py <output directory>")
        sys.exit(1)
    check_feas(sys.argv[1])

