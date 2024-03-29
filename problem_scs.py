#!/usr/bin/python3

from loguru import logger
from dataclasses import dataclass, field
import numpy as np
import sys
from typing import List
import re
import numpy as np

from ortools.sat.python import cp_model
from types import SimpleNamespace

from problem import Problem

UNKNOWN = cp_model.UNKNOWN
MODEL_INVALID = cp_model.MODEL_INVALID
FEASIBLE = cp_model.FEASIBLE
INFEASIBLE = cp_model.INFEASIBLE
OPTIMAL = cp_model.OPTIMAL


def add_parser_args(parser):
    pass
    # TODO
    # parser.add_argument('--tugs', type=int, default=2, help='Number of tugs to use')

@dataclass
class ScsInstance(object):
    tmax: int = 0
    jmax: int = 0
    smax: int = 0
    rmax: int = 0
    jsegs: int = 0
    pd: int = 0
    pw: int = 0
    mmin: int = 0
    mc: int = 0
    md: int = 0
    mw: int = 0
    aj: int = 0
    ej: int = 0
    dj: int = 0
    js: List[List[int]] = field(default_factory=lambda: [])

qre = re.compile(r'^.*\s*\{((\d+,?\s*)+)\},?\]?;?$')

def extract(ll):
    mm = qre.match(ll)
    return [int(ii) - 1 for ii in mm.groups()[0].split()]

def read_instance(inst_path):
    inst = ScsInstance()

    task_begin = []
    task_end = []
    task_duration = []
    qualified = []
    ll = None
    te = False
    tb = False
    td = False
    qual = False
    with open(inst_path, 'r') as fp:
        for line in fp:
            sl = line.strip().replace(';', '')
            if '=' in line:
                sp = sl.split('=')
                sp0 = sp[0].strip()
                if sp0 == 'NoTimeIntervals':
                    inst.tmax = int(sp[1])
                elif sp0 == 'NoSeafarers':
                    inst.smax = int(sp[1])
                elif sp0 == 'TaskBegin':
                    tb = True
                    task_begin.append(int(sp[1].split('[')[1]))
                    ll = task_begin
                elif sp0 == 'TaskEnd':
                    te = True
                    task_end.append(int(sp[1].split('[')[1]))
                    ll = task_end
                elif sp0 == 'TaskDuration':
                    td = True
                    task_duration.append(int(sp[1].split('[')[1]))
                    ll = task_duration
                elif sp0 == 'Qualified':
                    qual = True
                    qualified.append(extract(sl))
                elif sp0 == 'Day':
                    day = int(sp[1])
                elif sp0 == 'Week':
                    week = int(sp[1])
                elif sp0 == 'MinRestLength':
                    mrl = int(sp[1])
                elif sp0 == 'LongRestRequirement':
                    lrr = int(sp[1])
                elif sp0 == 'DayRestRequirement':
                    drr = int(sp[1])
                elif sp0 == 'WeekRestRequirement':
                    wrr = int(sp[1])
            elif sl.startswith(']'):
                tb = False
                te = False
                td = False
                qual = False
                ll = None
            elif ll is not None and (tb or te or td):
                if ']' in sl:
                    ll.append(int(sl[:-1]))
                    tb = False
                    te = False
                    td = False
                    ll = None
                else:
                    ll.append(int(sl))
            elif qual:
                qualified.append(extract(sl))

    inst.jmax = len(task_begin)
    # re arrange qual matrix from J -> S to S -> J
#     squal = [[] for ss in range(inst.smax)]
#     for jj in range(inst.jmax):
#         for sq in qualified[jj]:
#             squal[sq-1].append(jj + 1)

#     max_squal = max([len(qq) for qq in squal]) + 1
    max_jqual = max([len(jj) for jj in qualified]) + 1

    inst.pd = day
    inst.pw = week
    inst.mmin = mrl
    inst.mc = lrr
    inst.md = drr
    inst.mw = wrr
    inst.aj = [ii - 1 for ii in task_begin]
    inst.ej = [ii - 1 for ii in task_end]
    inst.dj = [ii - 1 for ii in task_duration]
    inst.js = qualified

    inst.jsegs = [inst.dj[jj] for jj in range(inst.jmax)]
    inst.rmax = int(2 * (inst.tmax / inst.md))

    return inst

class scs(Problem):
    def __init__(self, args, inst):
        super().__init__(args, inst.smax * inst.rmax)
        self.inst = inst

    def batch_evaluate(self, keys, threads=1, print_sol=False):
        super().batch_evaluate(keys)
        objs = np.zeros(len(keys), dtype=np.int64)

        for ii,kk in enumerate(keys):
            objs[ii] = self.assign_jobs(kk, print_sol)[0]
        return objs

    def assign_jobs(self, key, print_sol=False): # TODO impl print_sol
        """
        key structure: (TODO might need # of rests)
            1. Start of rest 1 [0,1] -> [0, tmax]
            2. ...
            3. Start of rest n (can't have more than (2 * tmax) / 3 rests (but probably need way less...
        Note that the model assumes there can always be a rest starting at 0 to avoid infeasibility

        Model idea: If we know when the rests start, then we can compute the necessary constraints for rest times (?) (and price them in the model)
        """
        pp = self.inst

        penalty = 0

        nrests = len(key) // pp.smax
        # prepend a 0 to the key, always have the option of resting at the start
        rstarts_ = np.around(key * pp.tmax, 0).astype(int)
        rstarts = np.reshape(rstarts_, (pp.smax, nrests))
        rstarts = np.hstack([np.zeros(pp.smax, dtype=int)[:,None], rstarts])
        nrests += 1

        model = cp_model.CpModel()
        xjs = {}
        xje = {}
        xjd = {}
        xji = {}
        xjp = {}
        xrs = {}
        xre = {}
        xrd = {}
        xri = {}
        xrp = {}
        yji = {}
        ssrt = {} # rest subtraction for rest[s,r] with time window ending at t
        zs = {}


        for ss in range(pp.smax):
            zs[ss] = model.NewBoolVar(f'zs_{ss}')

        for jj in range(pp.jmax):
            for ss in pp.js[jj]:
                for ii in range(pp.jsegs[jj]):
                    # Jobs
                    xjs[ss, jj, ii] = model.NewIntVar(pp.aj[jj], pp.ej[jj], f'xjs_{ss}_{jj}_{ii}')
                    xje[ss, jj, ii] = model.NewIntVar(pp.aj[jj], pp.ej[jj], f'xje_{ss}_{jj}_{ii}')
                    xjd[ss, jj, ii] = model.NewIntVar(0, pp.dj[jj], f'xjd_{ss}_{jj}_{ii}')
                    xjp[ss, jj, ii] = model.NewBoolVar(f'xjp_{ss}_{jj}_{ii}')
                    xji[ss, jj, ii] = model.NewOptionalIntervalVar(xjs[ss, jj, ii], xjd[ss, jj, ii],
                                                        xje[ss, jj, ii], xjp[ss, jj, ii], f'xji_{ss}_{jj}_{ii}')

        for ss in range(pp.smax):
            for rr in range(nrests):
                # Rests
                xrs[ss, rr] = model.NewIntVar(rstarts[ss, rr], rstarts[ss, rr], f'xrs_{ss}_{rr}')
                xre[ss, rr] = model.NewIntVar(rstarts[ss, rr], pp.tmax + 1, f'xre_{ss}_{rr}')
                xrd[ss, rr] = model.NewIntVar(0, pp.tmax - rstarts[ss,rr], f'xrd_{ss}_{rr}')
                xrp[ss, rr] = model.NewBoolVar(f'xrp_{ss}_{rr}')
                xri[ss, rr] = model.NewOptionalIntervalVar(xrs[ss, rr], xrd[ss, rr],
                                                           xre[ss, rr], xrp[ss, rr], f'xri_{ss}_{rr}')

                for tt in range(rstarts[ss, rr], pp.tmax):
                    ssrt[ss, rr, tt] = model.NewIntVar(0, pp.tmax - tt, f'ssrt_{ss}_{rr}_{tt}')

        # Covered now by adding an extra 0 start rest
        # # One extra rest for each seafarer at time 0, if necessary
        # for ss in range(pp.smax):
        #     xrs[ss] = model.NewIntVar(0, 0, f'xrs_{ss}')
        #     xre[ss] = model.NewIntVar(0, pp.tmax, f'xre_{ss}')
        #     xrd[ss] = model.NewIntVar(0, pp.tmax, f'xrd_{ss}')
        #     xrp[ss] = model.NewBoolVar(f'xrp_{ss}')
        #     xri[ss] = model.NewOptionalIntervalVar(xrs[ss], xrd[ss], xre[ss], xrp[ss], f'xri_{ss}')

        zs = {ss: model.NewBoolVar(f'z_{ss}') for ss in range(pp.smax)}

        # Constraints 5a: no overlapping segments of a given job
        for jj in range(pp.jmax):
            job_segs = [xji[ss, jj,ii] for ss in pp.js[jj] for ii in range(pp.jsegs[jj])]
            model.AddNoOverlap(job_segs)

        # Constraints 5b: no overlapping segments for a given seafarer
        for ss in range(pp.smax):
            job_segs = [xji[ss, jj, ii] for jj in range(pp.jmax) for ii in range(pp.jsegs[jj]) if ss in pp.js[jj]]
            rest_segs = [xri[ss, rr] for rr in range(nrests)]
            model.AddNoOverlap(job_segs + rest_segs)

        # Constraints 7
        for jj in range(pp.jmax):
            model.Add(sum(xjd[ss, jj, ii] for ss in pp.js[jj] for ii in range(pp.jsegs[jj]))
                    == pp.dj[jj])

        # Interval present if duration is non-zero
        for jj in range(pp.jmax):
            for ss in pp.js[jj]:
                for ii in range(pp.jsegs[jj]):
                    model.Add(xjd[ss, jj, ii] == 0).OnlyEnforceIf(xjp[ss, jj, ii].Not())
                    model.Add(xjd[ss, jj, ii] > 0).OnlyEnforceIf(xjp[ss, jj, ii])

        for ss in range(pp.smax):
            for rr in range(nrests):
                model.Add(xrd[ss, rr] == 0).OnlyEnforceIf(xrp[ss, rr].Not())
                model.Add(xrd[ss, rr] > 0).OnlyEnforceIf(xrp[ss, rr])

        # Connect seafarer on/off to their jobs
        for ss in range(pp.smax):
            model.Add(sum(xjp[ss, jj, ii] for jj in range(pp.jmax) for ii in range(pp.jsegs[jj]) if ss in pp.js[jj]) == 0).OnlyEnforceIf(zs[ss].Not())

        # Symmetry breaking on the segments
        for jj in range(pp.jmax):
            for ss in pp.js[jj]:
                for ii in range(pp.jsegs[jj] - 1):
                    # If segment i is not used, then neither is segment i+1
                    model.Add(xjp[ss, jj, ii+1] == 0).OnlyEnforceIf(xjp[ss, jj, ii].Not())
                    # Segment ordering only enforced if the second segment is turned on
                    model.Add(xje[ss, jj, ii] < xjs[ss, jj, ii+1]).OnlyEnforceIf(xjp[ss, jj, ii+1])

        # Rolling window rests (md/mw)
        for ((ss, rr, tt), svar) in ssrt.items():
            model.Add(svar >= 0).OnlyEnforceIf(xrp[ss, rr])
            model.Add(svar >= xre[ss, rr] - tt).OnlyEnforceIf(xrp[ss, rr])
        # Note: ssrt will never be made bigger than it needs to be since this
        # can only reduce the objective function value

        for ss in range(pp.smax):
            for tt in range(pp.tmax - pp.pd):
                # TODO have to handle the fact that the duration could exceed tt+md
                rel_rests = [xrd[ss, rr] - ssrt[ss, rr, max(pp.tmax - 1, tt + pp.pd)] for (rr, rs) in enumerate(rstarts[ss]) if rs >= tt and rs < tt + pp.pd]
                # rel_rests = [xrd[ss, rr] for (rr, rs) in enumerate(rstarts[ss]) if rs >= tt and rs < tt + pp.pd]
                debug = [(rr, rs) for (rr, rs) in enumerate(rstarts[ss]) if rs >= tt and rs < tt + pp.pd]
                if rel_rests:
                    model.Add(sum(rel_rests) >= pp.md)
                else:
                    penalty += 10 # TODO

        # Rests are at least mmin time units long
        for key, _ in xrd.items():
            if key[1] != 0:
                model.Add(xrd[key] >= 2).OnlyEnforceIf(xrp[key])

        model.Minimize(sum(zs[ss] for ss in range(pp.smax)))

        # print("starting solve")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.linearization_level = 0
        status = solver.Solve(model)
        # print(f"ending solve {status}")

        penobj = pp.smax * 3
        asgn = np.zeros((pp.smax, pp.tmax), dtype=int)
        if status == cp_model.OPTIMAL or status == FEASIBLE:
            obj = sum(solver.Value(zs[ss]) for ss in range(pp.smax))
            penobj = obj + penalty
            print(f"Solution found (obj: {obj}; penalized: {obj+penalty})")
            for jj in range(pp.jmax):
                # print(f"Job {jj+1}: ", end="")
                for ss in pp.js[jj]:
                    for ii in range(pp.jsegs[jj]):
                        vs = solver.Value(xjs[ss, jj, ii])
                        ve = solver.Value(xje[ss, jj, ii])
                        if solver.Value(xjp[ss, jj, ii]):
                            #print(f"s{ss}i_{ii}({vs},{ve},{solver.Value(xjd[ss,jj,ii])}) ", end="")
                            # print(f"s{ss}i_{ii}({vs},{ve}) ", end="")
                            for dd in range(solver.Value(xjd[ss, jj, ii])):
                                asgn[ss, vs + dd] = jj + 1
                # print()
            # print()
            # print()
            # for ss in range(pp.smax):
            #     print(f"Seafarer {ss}: ", end="")
            #     for tt in range(pp.tmax):
            #         val = asgn[ss, tt]
            #         if val >= 10:
            #             val = chr(ord('A') + (val - 10))
            #         print(val, end="")
            #     print()


        # elif status == INFEASIBLE:
        #     print(f"Model declared infeasible")
        # elif status == UNKNOWN:
        #     print("Couldn't find a feasible solution")
        # else:
        #     print("Model invalid")

        # Statistics.
        # print('\nStatistics')
        # print(f'  - conflicts      : {solver.NumConflicts()}')
        # print(f'  - branches       : {solver.NumBranches()}')
        # print(f'  - wall time      : {solver.WallTime():.2f} s')
        #print(f'  - solutions found: {solution_printer.solution_count()}' % )
        return (penobj, asgn)


if __name__ == "__main__":
    ss = scs(None, read_instance('../ship_crew_scheduling/data/2019_08_21_TestDaten.dat'))
    # ss = scs(None, read_instance('../ship_crew_scheduling/data/small.dat'))
    print(ss.inst.md)
    print(ss.inst.pd)
    lsp = np.linspace(0.1, 0.9, 25).tolist()
    key = np.array(lsp * ss.inst.smax)
    ss.assign_jobs(key)


