from collections import defaultdict
from loguru import logger
from scipy.spatial.distance import squareform
import argparse
import heapq
import itertools
import numpy as np
import pandas as pd
import sys
import time

from problem import Problem

def add_parser_args(parser):
#     parser.add_argument('--penalty', type=int, default=10, help='Penalty multiplier per unserviced pickup/delivery')
#     parser.add_argument('--no_mm34_output', action='store_true', default=False, help='Enables more compact output')
    pass

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
        obj1, start1, end1 = self.evaluate_fb(key, self.inst.prec, self.inst.succ, 0)
        if start1 is None:
            return obj1
        key = self.inst.horizon - end1
        obj2, start2, end2 = self.evaluate_fb(key, self.inst.succ, self.inst.prec, self.inst.jobs - 1)
        if start2 is None:
            return obj2
        key = self.inst.horizon - end2
        obj3, start3, end3 = self.evaluate_fb(key, self.inst.prec, self.inst.succ, 0)
        if start3 is None:
            return obj3
        if not print_sol:
            return min(obj1, obj2, obj3) + 1
        else:
            if obj1 <= obj3 and obj1 <= obj2:
                self.evaluate_fb(orig_key, self.inst.prec, self.inst.succ, 0, True)
                return obj1 + 1
            elif obj2 <= obj1 and obj2 <= obj3:
                key = self.inst.horizon - end1
                self.evaluate_fb(key, self.inst.succ, self.inst.prec, self.inst.jobs - 1, True)
                return obj2 + 1
            else:
                key = self.inst.horizon - end2
                self.evaluate_fb(key, self.inst.prec, self.inst.succ, 0, True)
                return obj3 + 1
        return obj3 + 1

    def evaluate_fb(self, key, prec, succ, start_at, print_sol=False):
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
        sg[start_at] = True
        dg = []
        dg.extend(succ[start_at])

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

            ef = np.max(end[prec[jstar]])

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
            for jj in succ[jstar]:
                if sg[prec[jj]].all():
                    dg.append(jj)

            # update resource consumption
            rremaining[start[jstar]:end[jstar]] -= inst.usage[jstar]

#             print(f"sg {sg}")
#             print(f"dg {dg}")
#             print(f"(start, end) {list(zip(start,end))}")
#             print(f"remaining {rremaining.T}")
#             print("====")

        if print_sol:
            logger.info(f"Job (start, end): {list(zip(start+1,end+1))}")
        return np.max(end), start, end





