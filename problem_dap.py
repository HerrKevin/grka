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
    parser.add_argument('--penalty', type=int, default=10, help='Penalty multiplier per unserviced pickup/delivery')
    parser.add_argument('--no_mm34_output', action='store_true', default=False, help='Enables more compact output')

def read_instance(inst_path):
    with open(inst_path, 'r') as fp:
        lines = [line.strip() for line in fp]
    inst = dap_instance()

    state = ''
    cc = 0
    val = []

    for line in lines:
        sp = line.split()
        if state == '':
            if line.startswith('customers'):
                inst.nn = int(sp[-1])
            elif line.startswith('max-time'):
                inst.max_time = int(sp[-1])
            elif line.startswith('vehicles'):
                inst.nvehicles = int(sp[-1])
            elif line.startswith('vehicle-capacity'):
                inst.vcap = [int(vv) for vv in sp[1:]]
            elif line.startswith('sym-transit-times'):
                state = 'stt'
            elif line.startswith('sym-transit-costs'):
                state = 'stc'
            elif line.startswith('time-windows'):
                state = 'tw'
            elif line.startswith('preferences'):
                state = 'pr'
        elif state == 'stt' or state == 'stc':
            cc += 1
            val.extend([int(ii) for ii in sp])
            if cc == 2 * inst.nn:
                if state == 'stt':
                    inst.tt = squareform(val)
                elif state == 'stc':
                    inst.tc = squareform(val)
                val = []
                state = ''
                cc = 0
        elif state == 'tw':
            if cc == 0:
                inst.ee = np.array([0] + [int(ii) for ii in sp])
                cc += 1
            elif cc == 1:
                inst.ll = np.array([inst.max_time] + [int(ii) for ii in sp])
                cc = 0
                state = ''
        elif state == 'pr':
            inst.pref.append([0] + [int(ii) for ii in sp])
            cc += 1
            if cc == inst.nvehicles:
                cc = 0
                state = ''
                inst.pref = np.array(inst.pref)
    return inst

class dap_instance(object):
    def __init__(self):
        self.nn = 0
        self.max_time = 0
        self.nvehicles = 0
        self.vcap = None
        self.tt = None
        self.tc = None
        self.ee = None
        self.ll = None
        self.pref = []

class dap(Problem):
    def __init__(self, args, inst):
        super().__init__(args, inst.nn)
        self.inst = inst

    def batch_evaluate(self, keys, threads=1, print_sol=False):
        super().batch_evaluate(keys)
        return np.array([self.evaluate(key, print_sol) for key in keys])

    def feasible(self, routes, arrival):
        for vv in range(inst.nvehicles):
            pass # TODO

    def fits_between(self, ra, first, second, btwn):
        inst = self.inst
        ea_btwn = ra + inst.tt[first, btwn]
        ea_second = max(ea_btwn, inst.ee[btwn]) + inst.tt[btwn, second]
        return ea_btwn <= inst.ll[btwn] and ea_second <= inst.ll[second]

    def can_pickup(self, route, rcap, idx, vcap):
        for nn in range(idx + 1, len(route)):
            if rcap[nn] + 1 > vcap:
                return False
        return True

    def evaluate(self, rkey, print_sol=False):
        key = rkey.copy()
        inst = self.inst

        use_pen = 2 * (self.args.penalty * np.std(inst.tc) + np.mean(inst.tc))

        vv = 0
        vborder = 1 / inst.nvehicles

        pd_sort = np.argsort(key)

        routes = [[0,0] for ii in range(inst.nvehicles)]
        rarrival = [[0, inst.max_time] for ii in range(inst.nvehicles)]
        rcosts = [0] * inst.nn
        rtime = [0] * inst.nn
        rcap = [[0,0] for ii in range(inst.nvehicles)] # num people in car after leaving node

        penalty = 0
        try_again = []

        for (reinsert, pd) in itertools.chain(zip([False] * len(pd_sort), pd_sort), try_again):
            pickup = pd + 1 # index 0 is the depot
            delivery = pd + 1 + inst.nn

            best_vv = -1
            best_p_idx = -1
            best_d_idx = -1
            best_delta = 1e99
            for vv in range(inst.nvehicles):
                for ii, (ff,tt) in enumerate(zip(routes[vv][:-1], routes[vv][1:])):
                    if self.can_pickup(routes[vv], rcap[vv], ii, inst.vcap[vv]) and \
                            self.fits_between(rarrival[vv][ii], ff, tt, pickup):
                        # Pickup insertion is feasible
                        delta = inst.pref[vv,pickup] - inst.tc[ff, tt] + inst.tc[ff, pickup] + inst.tc[pickup, tt]

                        # Now try to insert the delivery
                        for jj, (ffd, ttd) in enumerate(zip([pickup] + routes[vv][ii+1:-1], routes[vv][ii+1:]), start=ii):
                            ra = rarrival[vv][jj] if ffd != pickup else max(rarrival[vv][ii] + inst.tt[ff, pickup], inst.ee[pickup])
                            if self.fits_between(ra, ffd, ttd, delivery):
                                ddelta = inst.tc[ffd,delivery] + inst.tc[delivery, ttd]
                                if ffd != pickup:
                                    ddelta -= inst.tc[ffd, ttd]
                                if delta + ddelta < best_delta:
                                    best_vv = vv
                                    best_p_idx = ii
                                    best_d_idx = jj
                                    best_delta = delta + ddelta

#             print(f"Routes {routes}")
#             print(f"{best_vv}, {best_p_idx}, {best_d_idx}, {best_delta}")
            if best_vv >= 0:
                # Perform insertion
                routes[best_vv].insert(best_p_idx + 1, pickup)
                parr = max(rarrival[best_vv][best_p_idx] + inst.tt[routes[best_vv][best_p_idx], pickup], inst.ee[pickup])
                rarrival[best_vv].insert(best_p_idx + 1, parr)

                routes[best_vv].insert(best_d_idx + 2, delivery)
                darr = max(rarrival[best_vv][best_d_idx] + inst.tt[routes[best_vv][best_d_idx], pickup], inst.ee[pickup])
                rarrival[best_vv].insert(best_d_idx + 2, darr)

                rcosts[best_vv] += best_delta

#                 print(f"\nv: {best_vv}; p idx: {best_p_idx}; d idx: {best_d_idx}")
#                 print(f"Before update: {rcap}")
                rcap[best_vv].insert(best_d_idx + 1, rcap[best_vv][best_d_idx])
#                 print(f"(D) Insert {rcap[best_vv][best_d_idx]} at {best_d_idx + 1}")
#                 print(f"(D) update: {rcap}")
                rcap[best_vv].insert(best_p_idx + 1, rcap[best_vv][best_p_idx] + 1)
#                 print(f"(P) Insert {rcap[best_vv][best_p_idx]} at {best_p_idx + 1}")
#                 print(f"(P) update: {rcap}")
                for ii in range(best_p_idx + 2, best_d_idx + 2):
                    rcap[best_vv][ii] += 1
#                 print(f"After: {rcap}\n")
            elif not reinsert:
                # Try to put this pd back in once the routes are complete
                # allowing for shifting of times
                # try_again.append((True, pd)) # TODO TODO TODO for now just add the penalty
                penalty += use_pen
            else:
                logger.critical(":-(") # apply a penalty...
                penalty += use_pen

#             print(f"Routes after {routes}")
#             print(f"rarrival after {rarrival}")
#             print(f"rcap after {rcap}")

        if print_sol:
            if self.args.no_mm34_output:
                if penalty > 0:
                    logger.critical("Infeasible solution follows! (penalty: {penalty})")
                for vv in range(inst.nvehicles):
                    logger.info(f"Vehicle {vv} (route): {routes[vv]}")
                    logger.info(f"Vehicle {vv} (arr.): {rarrival}")
            else:
                if penalty > 0:
                    print("###RESULT: Timeout.")
                else:
                    print(f"###RESULT: Feasible.")
                    print(f"###COST: {np.sum(rcosts)}")
                    for vv in range(inst.nvehicles):
                        froute = " ".join([str(val) for val in routes[vv]])
                        print(f"###VEHICLE {vv}: {froute}")
                    print(f"###CPU-TIME: {time.process_time():.2f}")

        return np.sum(rcosts) + penalty





