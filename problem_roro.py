#!/usr/bin/python3

import argparse
import numpy as np
import pandas as pd
from pqdict import pqdict
from collections import defaultdict
import os.path
import sys

from problem import Problem

def add_parser_args(parser):
    parser.add_argument('--tugs', type=int, default=2, help='Number of tugs to use')

def read_instance(inst_path):
    if not os.path.exists(inst_path):
        logger.error(f"Instance path does not exist: {inst_path}")
        sys.exit(1)

    if inst_path.endswith('.dzn'):
        inst = parse_instance_dzn(inst_path)
    else:
#         if args.tugs is None:
#             print("A xls/xlsx detected. --tugs must be specified!")
#             sys.exit(1)
#         inst = parse_instance_xlsx(inst_path) if not args.labeled else parse_instance_xlsx_labeled(inst_path)
        inst = parse_instance_xlsx_labeled(inst_path)
    return inst

def parse_instance_dzn(path):
    """
    Note that this is not meant for reading general dzn files... just those
    generated with the scripts in this project
    """
    with open(path, 'r') as fp:
        lines = [line.strip() for line in fp]

    inst = roro_instance()
    read_matrix = False
    target = None
    ii = 0
    for line in lines:
        sp = line.split('=')
        if read_matrix:
            if line.startswith("|];"):
                read_matrix = False
                ii = 0
            else:
                vals = parse_matrix_line(sp[0].replace("[","").strip())
                for jj, vv in enumerate(vals):
                    target[ii,jj] = vv
                ii += 1
        elif line.startswith("nship"):
            inst.nship = int(sp[-1][:-1])
        elif line.startswith("nquay"):
            inst.nquay = int(sp[-1][:-1])
        elif line.startswith("k"):
            inst.kk = int(sp[-1][:-1])
        elif line.startswith("PSS"):
            read_matrix = True
            inst.pss = np.zeros((inst.nship, inst.nship))
            target = inst.pss
            vals = parse_matrix_line(sp[1].replace("[","").strip())
            for jj, vv in enumerate(vals):
                target[ii,jj] = vv
            ii += 1
        elif line.startswith("PQQ"):
            read_matrix = True
            inst.pqq = np.zeros((inst.nquay, inst.nquay))
            target = inst.pqq
            vals = parse_matrix_line(sp[1].replace("[","").strip())
            for jj, vv in enumerate(vals):
                target[ii,jj] = vv
            ii += 1
        elif line.startswith("PSQ"):
            read_matrix = True
            inst.psq = np.zeros((inst.nship, inst.nquay))
            target = inst.psq
            vals = parse_matrix_line(sp[1].replace("[","").strip())
            for jj, vv in enumerate(vals):
                target[ii,jj] = vv
            ii += 1
    return inst

def parse_instance_xlsx_labeled(path):
    xlf = pd.read_excel(path, sheet_name=None)
    sdf = xlf['position_sequence']
    ldf = xlf['precedence_loading']
    udf = xlf['precedence_unloading']

    slots = {}
    slotrev = {}
    for row in sdf.itertuples():
        slots[row.position_onboard] = int(row.dis_seq) - 1
        slotrev[int(row.dis_seq) - 1] = row.position_onboard

    inst = roro_instance()
    inst.nship = max(slots.values()) + 1
    inst.nquay = max(slots.values()) + 1

    inst.slots = slots
    inst.slotrev = slotrev
    inst.labeled = True

    inst.psq = np.zeros((inst.nship, inst.nquay))
    np.fill_diagonal(inst.psq, 1)
    inst.dpsq = defaultdict(list)
    inst.drpsq = defaultdict(list)

    for ii in range(inst.nship):
        inst.dpsq[ii].append(ii)
        inst.drpsq[ii].append(ii)

    inst.pss = np.zeros((inst.nship, inst.nship))
    inst.pqq = np.zeros((inst.nquay, inst.nquay))

    inst.dpss = defaultdict(list)
    inst.dpqq = defaultdict(list)
    inst.drpss = defaultdict(list)
    inst.drpqq = defaultdict(list)

    for row in ldf.itertuples():
        suc = slots[row.suc]
        pred = slots[row.pred]
        inst.pss[suc,pred] = 1

        inst.dpss[suc].append(pred)
        inst.drpss[pred].append(suc)

    for row in udf.itertuples():
        suc = slots[row.suc]
        pred = slots[row.pred]
        inst.pqq[suc,pred] = 1

        inst.dpqq[suc].append(pred)
        inst.drpqq[pred].append(suc)

    return inst


class roro_instance(object):
    def __init__(self):
        self.nship = None
        self.nquay = None
        self.kk = None
        self.pss = None
        self.pqq = None
        self.psq = None

        # Precedence mapping ([i] is blocked by [j, k, ...])
        self.dpss = None
        self.dpqq = None
        self.dpsq = None

        # Reverse precedence mapping ([i] blocks [j, j, ...])
        self.drpss = None
        self.drpqq = None
        self.drpsq = None

def vd(dd,ll): # vec dict
    if len(ll) == 0:
        return []
    else:
        arr = np.vectorize(dd.get)(np.array(ll))
        arr.sort()
        return arr

class roro(Problem):
    def __init__(self, args, inst):
        super().__init__(args, inst.nship + inst.nquay)
        self.inst = inst
        self.inst.kk = args.tugs

    def batch_evaluate(self, keys, threads=1, print_sol=False):
        super().batch_evaluate(keys)
        # TODO multiprocessing map? Or thread myself to avoid pickle overhead
        objs = np.zeros(len(keys), dtype=np.int64)
        for ii,kk in enumerate(keys):
            objs[ii] = self.greedy(kk, print_sol)[0]
        return objs

    def greedy(self, key, print_sol=False):
        """
        Assumes:
            1. The maximum number of vehicles on the ship at one time is floor(inst.kk / 2)
            2. The trailer slots on the vessel are topologically sorted from aft to fore, starboard to port
        """
        out = ""
        inst = self.inst

        key /= 1.0001 # we can't have any entries equal to 1.0 or the pqueue will fail

        # Heap of dependency counts for trailers on ship
        heaps = pqdict({fore: (len(aa) + key[fore], fore) for (fore,aa) in inst.drpss.items()})
        for ii in range(inst.nship):
            if ii not in inst.drpss:
                heaps.additem(ii, (0, ii))
#
        # Heap of dependency counts for trailers on quay
        # + 1 because it is assumed all trailer slots on the ship are blocked
        heapq = pqdict({aft: (len(ff) + 1 + key[inst.nship + aft], aft) for (aft,ff) in inst.drpqq.items()})
        for ii in range(inst.nquay):
            if ii not in inst.drpqq:
                heapq.additem(ii, (1 + key[inst.nship + ii], ii))

        max_time = (inst.nship + inst.nquay) + inst.kk
        wsq = [0] * max_time # vehicles from ship to quay
        wqs = [0] * max_time # vehicles from quay to ship
        wqq = [0] * max_time # vehicles waiting at quay
        # for each trailer on the ship, the time period in which the trailer leaves the ship
        xx = [-1] * inst.nship
        # for each trailer on the quay, the time period in which the trailer leaves the quay
        yy = [-1] * inst.nquay
#
        # The first time period always involves moving half the vehicles to the
        # ship, and the other half doing nothing
#         assert(inst.kk % 2 == 0)
        hv = inst.kk // 2
        wqs[0] = hv
        wqq[0] = hv
#
        if print_sol:
            out += f"1 unloaded {{}} loaded {{}} (# not unloaded: {len(heaps)}; # not loaded: {len(heapq)}) moving {inst.kk/2} tugs on to ship\n"
        tt = 1
        while (len(heaps) > 0 or len(heapq) > 0) and tt < max_time:
            unloaded = []
            wsq[tt] = hv
            for iiu in range(min(len(heaps), wqs[tt - 1])):
                unload, (pval, ignore) = heaps.topitem()
                if int(pval) == 0:
                    heaps.pop()
                    unloaded.append(unload)
                    xx[unload] = tt
                    # TODO if we do the update of heaps here, we can unload trailers
                    # that are blocked by another unloaded container. Is this good?
                else:
                    wsq[tt] -= 1
                    wqq[tt] += 1

            if len(heapq) > 0:
                wqs[tt] = min(len(heapq), hv)

            for uu in unloaded:
                # Free place on ship for this trailer
                heapq.updateitem(uu, (heapq[uu][0] - 1, uu))
                # Trailers on ship blocked by this trailer
                for blocking in inst.dpss[uu]:
                    heaps.updateitem(blocking, (heaps[blocking][0] - 1, blocking))

            # count how many loaded trailers have the same destination slot as
            # loaded trailers (only max. n-1 trailers can be the same)
            n_in_unloaded = 0

            add_again = False

            loaded = []
            for iil in range(min(len(heapq), wqq[tt - 1] + wsq[tt - 1])):
                load, (pval, ignore) = heapq.topitem()
                if load in unloaded:
    #                 print(f"Req. Load: {load+1}; Unloading: {np.array(unloaded)+1}")
                    n_in_unloaded += 1
                if n_in_unloaded == hv: # Have to pick a different trailer; slot blocked
    #                 print(f"n_in_unloaded == hv at {tt}")
                    heapq.pop()
                    nload, (npval, nignore) = heapq.topitem()
    #                 print(f"Choose: {load+1}, ({pval}, {ignore}); Alternative: {nload+1}, ({npval}, {nignore})")
                    add_again = True
                    taload, (taval, taignore) = load, (pval, ignore)
                    load, (pval, ignore) = nload, (npval, nignore)

                if int(pval) == 0:
                    heapq.pop()
                    loaded.append(load)
                    yy[load] = tt

                if add_again:
                    heapq.additem(taload, (taval, taignore))
                    add_again = False

            for ll in loaded:
                for inback in inst.dpqq[ll]:
                    heapq.updateitem(inback, (heapq[inback][0] - 1, inback))

            #if inst.labeled:
            if print_sol:
                out += f"{tt+1} unloaded {vd(inst.slotrev, unloaded)} loaded {vd(inst.slotrev, loaded)} (# not unloaded: {len(heaps)}; # not loaded: {len(heapq)})\n"
            #else:
            #    out += f"{tt} unloaded {np.array(unloaded) + 1} loaded {np.array(loaded) + 1} (# not unloaded: {len(heaps)}; # not loaded: {len(heapq)})\n"

            tt += 1
        if print_sol:
            print(out)
        if tt >= max_time:
            return max_time * 10,None,None,None,None,None,"infeasible or error"
        wsq[tt] = wqs[tt-1] # Remove tugs from ship, if any are left on there

        return tt,xx,yy,wsq,wqs,wqq,out



