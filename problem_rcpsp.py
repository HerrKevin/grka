from collections import defaultdict
from loguru import logger
from scipy.spatial.distance import squareform
import argparse
import copy
import heapq
import itertools
import numpy as np
import pandas as pd
import sys
import time

from problem import Problem

def add_parser_args(parser):
    parser.add_argument('--mod_makespan_l', type=int, default=0, help='L value for the modified makespan from Mendes et al. (2007). A value of 0 turns off the modified makespan.')

def read_instance(inst_path):
    with open(inst_path, 'r') as fp:
        lines = [line.strip() for line in fp]

    inst = rcpsp_instance()

    prec_info = False
    rec_info = False
    rcounter = 0

    for line in lines:
        csp = line.split(':')
        ssp = line.split()
        if line.startswith('jobs'):
            inst.jobs = int(csp[1])
            inst.usage = [[] for ii in range(inst.jobs)]
            inst.duration = np.zeros(inst.jobs, dtype=np.int)
        elif line.startswith('horizon'):
            inst.horizon = int(csp[1])
        elif inst.resources == 0 and 'renewable' in line:
            inst.resources = int(csp[1].split()[0])
        elif 'modes' in line:
            prec_info = True
        elif prec_info:
            if line.startswith('***'):
                prec_info = False
            else:
                succ = [int(suc) - 1 for suc in ssp[3:]]
                pre = int(ssp[0]) - 1
                inst.succ[pre].extend(succ)
                for suc in succ:
                    inst.prec[suc].append(pre)
        elif line.startswith('----'):
            rec_info = True
        elif rec_info:
            if line.startswith('***'):
                rec_info = False
            else:
                jj = int(ssp[0]) - 1
                inst.usage[jj] = [int(rr) for rr in ssp[3:]]
                inst.duration[jj] = int(ssp[2])
        elif line.startswith('RESOURCEAVAIL'):
            rcounter += 1
        elif rcounter > 0:
            if rcounter == 2:
                inst.rcap = np.array([int(rr) for rr in ssp])
            rcounter += 1
    logger.info(f"Jobs: {inst.jobs}")
    logger.info(f"Horizon: {inst.horizon}")
    logger.info(f"Resources: {inst.resources}")
    logger.info(f"Prec: {inst.prec}")
    logger.info(f"Succ: {inst.succ}")
    logger.info(f"Duration: {inst.duration}")
    logger.info(f"Usage: {inst.usage}")
    logger.info(f"Capacity: {inst.rcap}")
    inst.usage = np.array(inst.usage)
    return inst

class rcpsp_instance(object):
    def __init__(self):
        self.jobs = 0
        self.horizon = 0
        self.resources = 0
        self.prec = defaultdict(list)
        self.succ = defaultdict(list)
        self.duration = []
        self.usage = None
        self.rcap = []


class rcpsp(Problem):
    def __init__(self, args, inst):
        super().__init__(args, inst.jobs)
        self.inst = inst
        # Keys: priority of each job

        # precompute lf
        self.lf = np.zeros(inst.jobs, dtype=np.int)
        self.lf[-1] = inst.horizon
        oopen = inst.prec[inst.jobs - 1]
        while oopen:
            new_open = []
            for oo in oopen:
                self.lf[oo] = np.min(self.lf[inst.succ[oo]] - inst.duration[inst.succ[oo]])
                new_open.extend(inst.prec[oo])
            oopen = new_open


    def batch_evaluate(self, keys, threads=1, print_sol=False):
        super().batch_evaluate(keys)
        return np.array([self.evaluate(key, print_sol) for key in keys])

    def evaluate(self, key, print_sol=False):
        """
        FBI decoder as suggested by "A biased random-key genetic
        algorithm with forward-bckward improvement for the resource constrained
        project scheduling problem" by Goncalves, Resende and Mendes

        Note: add 1 to the objective function because time should start at 1 and not 0

        """
        orig_key = np.copy(key)
        obj1, start1, end1 = self.evaluate_initial(key)
        if start1 is None:
            return obj1

        obj2, start2, end2 = self.backwards(start1, end1)

        obj3, start3, end3 = self.forwards(start2, end2)

        if print_sol:
            logger.info(f"Job (start, end): {list(zip(start3+1,end3+1))}")

        if not print_sol and self.args.mod_makespan_l > 0:
            return obj3 + self.modify_makespan(start3, end3) + 1
        else:
            return obj3 + 1

    def modify_makespan(self, start, end):
        ll = self.args.mod_makespan_l
        inst = self.inst

        numer = end[-1]
        denom = end[-1]

        inspect = [(1, pp) for pp in inst.prec[inst.jobs - 1]]
        while inspect:
            (dist, jj) = inspect.pop()
            if dist > ll:
                continue

            if ll + 1 <= ll:
                inspect.extend([(ll+1, pp) for pp in inst.prec[jj]])

            numer += end[jj]
            denom += end[-1]

        return numer / denom

    def evaluate_initial(self, key):
        """
        Decoder from page 474 (Fig. 6) from "A biased random-key genetic
        algorithm with forward-bckward improvement for the resource constrained
        project scheduling problem" by Goncalves, Resende and Mendes
        """
        inst = self.inst

        # precompute latest finish times with backwards recursion

        start = np.zeros(inst.jobs, dtype=int)
        end = np.zeros(inst.jobs, dtype=int)

        sg = np.zeros(inst.jobs, dtype=bool)
        sg[0] = True
        dg = []
        dg.extend(inst.succ[0])

        rremaining = np.zeros((inst.horizon, inst.resources), dtype=int)
        rremaining[:] = inst.rcap

#         print("======== get it going ===========")
#         print(sg)
#         print(dg)
#         print(inst.succ)

        for gg in range(inst.jobs - 1):
            # select highest priority eligble job
#             if not dg:
#                 print(f"sg {sg}")
#                 print(f"dg {dg}")
#                 print(f"(start, end) {list(zip(start,end))}")
#                 print(f"remaining {rremaining.T}")
#                 sys.exit(1)

            jstar_idx = np.argmax(key[dg])
            jstar = dg[jstar_idx]
#             print(f"jstar: {jstar}")
            del dg[jstar_idx]

            ef = np.max(end[inst.prec[jstar]])

            earliest_start = ef
            latest_finish = self.lf[jstar] - inst.duration[jstar]

            job_end = -1
            ravail = (rremaining[earliest_start:latest_finish] >= inst.usage[jstar]).all(axis=1)
            run = 0
            for tt, avail in enumerate(ravail):
                if run == inst.duration[jstar]:
                    job_end = tt + earliest_start
                    break
                if avail:
                    run += 1
                else:
                    run = 0

#             print(f"{jstar} ravail {ravail}")
#             print(earliest_start, latest_finish)
            if job_end < 0:
                # Infeasible sorting; not sure what to do other than to give up
#                 logger.error("need to handle an infeasible sorting")
                return 1e50, None, None

            # TODO I'm ignoring the gamma part of Fig. 6, but including it mihgt
            # speed this up.. or not. not sure.
            extra = 1 if inst.duration[jstar] > 0 else 0
            start[jstar] = job_end - inst.duration[jstar] + extra
            end[jstar] = job_end
            sg[jstar] = True

            # update dg with jobs that can run
            for jj in inst.succ[jstar]:
                if sg[inst.prec[jj]].all():
                    dg.append(jj)

            # update resource consumption
            rremaining[start[jstar]:end[jstar]] -= inst.usage[jstar]

#             print(f"sg {sg}")
#             print(f"dg {dg}")
#             print(f"(start, end) {list(zip(start,end))}")
#             print(f"remaining {rremaining.T}")
#             print("====")
        return end[-1], start, end

    def backwards(self, start, end):
        inst = self.inst

#         pend = np.copy(end)

        nstart = np.copy(start)
        nend = np.copy(end)
#         print(list(zip(nstart, nend)))

        nrremaining = np.zeros((inst.horizon, inst.resources), dtype=int)
        nrremaining[:] = inst.rcap

        jj = inst.jobs - 1
        end_max = np.max(end)
        nstart[jj] = end_max
        nend[jj] = end_max

        inspect = list(range(inst.jobs - 1))
        for gg in range(inst.jobs - 1):
            # Select last scheduled job that we haven't looked at yet
            iidx = np.argmax(end[inspect])
            jj = inspect.pop(iidx)

            if inst.succ[jj]:
                earliest_succ = np.min(nstart[inst.succ[jj]]) - 1
            else:
                earliest_succ = end_max

            iu = inst.usage[jj]
            for tt in range(earliest_succ, 0, -1):
                extra = 1 if inst.duration[jj] > 0 else 0
                ss = tt - inst.duration[jj] + extra
                if (nrremaining[ss:tt+1,:] >= iu).all():
                    # Job fits; move it back here
                    nstart[jj] = ss
                    nend[jj] = tt
                    nrremaining[ss:tt+1,:] -= iu
                    break

#         print(list(zip(nstart, nend)))
        return nend[-1], nstart, nend

    def forwards(self, start, end):
        inst = self.inst

        pend = np.copy(end)

        nstart = np.copy(start)
        nend = np.copy(end)
#         print(list(zip(nstart, nend)))

        nrremaining = np.zeros((inst.horizon, inst.resources), dtype=int)
        nrremaining[:] = inst.rcap

        jj = 0
        nstart[jj] = 0
        nend[jj] = 0

        inspect = list(range(1, inst.jobs))
        for gg in range(inst.jobs - 1):
            # Select last scheduled job that we haven't looked at yet
            iidx = np.argmin(start[inspect])
            jj = inspect.pop(iidx)

            if inst.prec[jj]:
                latest_prec = np.max(nend[inst.prec[jj]])
            else:
                latest_prec = 0

            iu = inst.usage[jj]
            for ss in range(latest_prec+1, inst.horizon):
                extra = 1 if inst.duration[jj] > 0 else 0
                tt = ss + inst.duration[jj] - extra # tt is included as a resource consuming time
                if (nrremaining[ss:tt+1,:] >= iu).all():
                    # Job fits; move it back here
                    nstart[jj] = ss
                    nend[jj] = tt
                    nrremaining[ss:tt+1,:] -= iu
                    break

#         print(list(zip(nstart, nend)))
        return nend[-1], nstart, nend







