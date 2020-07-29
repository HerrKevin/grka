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
        super().__init__(args, inst.nn * 3)
        # Keys:
        # 0:inst.nn; PD order
        # inst.nn:inst.nn*3; position in feasible visit window for pickups and deliveries
        self.inst = inst
        self.pen = 2 * (self.args.penalty * np.std(inst.tc) + np.mean(inst.tc))
        logger.info(f"Using penalty {self.pen} per infeasibility.")

    def batch_evaluate(self, keys, threads=1, print_sol=False):
        super().batch_evaluate(keys)
        return np.array([self.evaluate(key, print_sol) for key in keys])

    def feasible(self, routes, arrival):
        for vv in range(inst.nvehicles):
            pass # TODO

#     def fits_between(self, ra, first, second, btwn):
#         inst = self.inst
#         ea_btwn = ra + inst.tt[first, btwn]
#         ea_second = max(ea_btwn, inst.ee[btwn]) + inst.tt[btwn, second]
#         return ea_btwn <= inst.ll[btwn] and ea_second <= inst.ll[second]
    def fits_between(self, ff, ffa, tt, tta, btwn):
        inst = self.inst

        arr_btwn = max(ffa + inst.tt[ff,btwn], inst.ee[btwn])
        if arr_btwn <= inst.ll[btwn] and arr_btwn + inst.tt[btwn,tt] <= tta:
            return True
        return False

    def can_pickup(self, route, rcap, idx, vcap):
        for nn in range(idx + 1, len(route)):
            if rcap[nn] + 1 > vcap:
                return False
        return True

    def evaluate(self, rkey, print_sol=False):
        inst = self.inst
        key = rkey[:inst.nn] # PD ordering
        bkey = rkey[inst.nn:] # Placement of a delivery/pickup within the time frame available


        vv = 0
        vborder = 1 / inst.nvehicles

        pd_sort = np.argsort(key)

        routes = [[0,0] for ii in range(inst.nvehicles)]
        rarrival = [[0, inst.max_time] for ii in range(inst.nvehicles)]
        rcosts = [0] * inst.nvehicles
        rcap = [[0,0] for ii in range(inst.nvehicles)] # num people in car after leaving node
        prefs = 0

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
                for ii, (ff,tt,ffa,tta) in enumerate(zip(routes[vv][:-1], routes[vv][1:], rarrival[vv][:-1], rarrival[vv][1:])):
                    if rcap[vv][ii] + 1 < inst.vcap[vv] and self.fits_between(ff, ffa, tt, tta, pickup):
#                             self.fits_between(rarrival[vv][ii], ff, tt, pickup):
                        # Pickup insertion is feasible
                        delta = inst.tc[ff, pickup] + inst.tc[pickup, tt] - inst.tc[ff, tt]

                        # Now try to insert the delivery
                        parr = max(ffa + inst.tt[ff, pickup], inst.ee[pickup])
                        for jj, (ffd, ttd, ffda, ttda) in enumerate(zip([pickup] + routes[vv][ii+1:-1], routes[vv][ii+1:],
                                                                        [parr] + rarrival[vv][ii+1:-1], rarrival[vv][ii+1:]), start=ii):
                            if rcap[vv][jj] + 1 > inst.vcap[vv]:
                                break # no point in looking at more delivery options; we would exceed the capacity after this node
                            if self.fits_between(ffd, ffda, ttd, ttda, delivery):
                                ddelta = inst.tc[ffd, delivery] + inst.tc[delivery, ttd]
                                if ffd == pickup:
                                    # adjust for the fact that the pickup now goes to the delivery rather than the node on the path
                                    ddelta -= inst.tc[pickup, tt]
                                else:
                                    ddelta -= inst.tc[ffd, ttd]
                                if delta + ddelta < best_delta:
                                    best_vv = vv
                                    best_p_idx = ii
                                    best_d_idx = jj
                                    best_delta = delta + ddelta

            if best_vv >= 0:
                # Perform insertion
                routes[best_vv].insert(best_p_idx + 1, pickup)
                parr_min = max(rarrival[best_vv][best_p_idx] + inst.tt[routes[best_vv][best_p_idx], pickup], inst.ee[pickup])

                parr_max = min(inst.ll[pickup], rarrival[best_vv][best_p_idx + 1]
                               - inst.tt[pickup, routes[best_vv][best_p_idx + 1]])

                rarrival[best_vv].insert(best_p_idx + 1, parr_min + (parr_max - parr_min) * bkey[pickup - 1])

                routes[best_vv].insert(best_d_idx + 2, delivery)

                darr_min = max(rarrival[best_vv][best_d_idx + 1] + inst.tt[routes[best_vv][best_d_idx + 1], delivery], inst.ee[delivery])

                darr_max = min(inst.ll[delivery], rarrival[best_vv][best_d_idx + 2]
                               - inst.tt[delivery, routes[best_vv][best_d_idx + 2]])

                rarrival[best_vv].insert(best_d_idx + 2,  darr_min + (darr_max - darr_min) * bkey[delivery - 1])

                rcosts[best_vv] += best_delta

                rcap[best_vv].insert(best_d_idx + 1, rcap[best_vv][best_d_idx])
                rcap[best_vv].insert(best_p_idx + 1, rcap[best_vv][best_p_idx] + 1)
                for ii in range(best_p_idx + 2, best_d_idx + 2):
                    rcap[best_vv][ii] += 1
                prefs += inst.pref[best_vv][pickup]
            elif not reinsert:
                # Try to put this pd back in once the routes are complete
                # allowing for shifting of times
                # try_again.append((True, pd)) # TODO TODO TODO for now just add the penalty
                penalty += self.pen
            else:
                logger.critical(":-(") # apply a penalty...
                penalty += self.pen

        if print_sol:
            if self.args.no_mm34_output:
                if penalty > 0:
                    logger.critical(f"Infeasible solution follows! (penalty: {penalty})")
                for vv in range(inst.nvehicles):
                    logger.info(f"Vehicle {vv} (route): {routes[vv]}")
                    logger.info(f"Vehicle {vv} (arr.): {rarrival[vv]}")
#                     for (ff,tt,ffa,tta) in zip(routes[vv][:-1], routes[vv][1:], rarrival[vv][:-1], rarrival[vv][1:]):
#                         print(f"Drive from {ff}@{ffa} to {tt}@{tta}; {inst.tt[ff,tt]} ok? {ffa + inst.tt[ff,tt] <= tta}")
            else:
                if penalty > 0:
                    print("###RESULT: Timeout.")
                else:
                    print(f"###RESULT: Feasible.")
                    print(f"###COST: {np.sum(rcosts) + prefs}")
                    for vv in range(inst.nvehicles):
                        froute = " ".join([str(val) for val in routes[vv][1:-1]])
                        print(f"###VEHICLE {vv+1}: {froute}")
                    print(f"###CPU-TIME: {time.process_time():.2f}")

#         print(f"# infeasible PDs: {penalty / self.pen}")
        return np.sum(rcosts) + prefs + penalty





