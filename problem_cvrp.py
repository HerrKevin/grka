import argparse
import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.distance import euclidean
import sys

from problem import Problem

def add_parser_args(parser):
    parser.add_argument('--max_k_opt', type=int, default=3, help='Maximum k-opt improvements per vehicle')

def read_instance(inst_path):
    with open(inst_path, 'r') as fp:
        lines = [line.strip() for line in fp]

    inst = cvrp_instance()

    node_coord = False
    demand = False
    for line in lines:
        if line.startswith('CAPACITY'):
            inst.cap = int(line.split(":")[-1])
        elif line.startswith('NODE_COORD_SECTION'):
            node_coord = True
        elif line.startswith('DEMAND_SECTION'):
            node_coord = False
            demand = True
        elif line.startswith('DEPOT_SECTION'):
            demand = False
        elif node_coord:
            sp = [int(ii) for ii in line.split()]
            inst.coords.append(sp[1:])
        elif demand:
            inst.dmd.append(int(line.split()[-1]))

    inst.coords = np.array(inst.coords)
    logger.info(f"Capacity: {inst.cap}")
    logger.info(f"Node coordinates: {inst.coords.T}")
    logger.info(f"Demand: {inst.dmd}")

    return inst

class cvrp_instance(object):
    def __init__(self):
        self.cap = 0
        self.coords = []
        self.dmd = []

class cvrp(Problem):
    def __init__(self, args, inst):
        super().__init__(args, inst.coords.shape[0])
        self.inst = inst

        # dist hash
        self.dists = {}

    def batch_evaluate(self, keys, threads=1, print_sol=False):
        super().batch_evaluate(keys)
        return np.array([self.evaluate(key, print_sol) for key in keys])

    def dist(self, ff, tt):
        inst = self.inst
        if (ff, tt) not in self.dists:
            self.dists[ff,tt] = euclidean(inst.coords[ff], inst.coords[tt])
        return self.dists[ff,tt]

    def evaluate(self, key, print_sol=False):
        dist = self.dist
        inst = self.inst
        obj = 0
        vpaths = [[0]]
        load = 0
        curpath = 0
        depot = 0

        for ii, vv in enumerate(np.argsort(key)):
            prev = vpaths[curpath][-1]
            dnext = dist(prev, vv)

            # cheaper to go to the depot and start a new route than make this transit?
            dist_p_d_d_c = dist(prev, depot) + dist(depot, vv)
            if dist_p_d_d_c < dnext:
                obj += dist_p_d_d_c
                vpaths[curpath].append(0)
                curpath += 1
                load = 0
                vpaths.append([0])

            if load + inst.dmd[vv] < inst.cap:
                load += inst.dmd[vv]
                obj += dnext
                vpaths[curpath].append(vv)
            else:
                obj += dist(prev, depot) + dist(depot, vv)
                vpaths[curpath].append(0)
                curpath += 1
                load = inst.dmd[vv]
                vpaths.append([0, vv])

        if curpath != len(vpaths) and vpaths[curpath][-1] != 0:
            obj += dist(vpaths[-1][-1], depot)
            vpaths[curpath].append(0)

        if vpaths[-1] == [0,0]:
            del vpaths[-1]

        # Local improvement step; try to improve things with 2-opt
        for kk, vpath in enumerate(vpaths):
            best_delta = -1
            best_swap = None
            cc = 0
            while best_delta < 0 and cc < self.args.max_k_opt:
                best_delta = 0
                best_swap = None
                for ii, v1 in enumerate(vpath[1:-1], 1):
                    previ = vpath[ii-1]
                    posti = vpath[ii+1]
                    for jj, v2 in enumerate(vpath[ii+1:-1], ii+1):
                        prevj = vpath[jj-1]
                        postj = vpath[jj+1]
                        delta = 0
                        if jj == ii + 1:
                            delta = dist(previ, v2) + dist(v2, v1) + dist(v1, postj)
                            delta -= dist(previ, v1) + dist(v1, v2) + dist(v2, postj)
                        else:
                            delta = dist(previ, v2) + dist(v2, posti) + dist(prevj, v1) + dist(v1, postj)
                            delta -= dist(previ, v1) + dist(v1, posti) + dist(prevj, v2) + dist(v2, postj)
                        if delta - best_delta < -1e-4:
                            best_delta = delta
                            best_swap = (ii, jj)
                if best_delta < 0:
                    ii,jj = best_swap
                    obj += best_delta # best_delta is negative
                    vpath[ii],vpath[jj] = vpath[jj], vpath[ii]
                    cc += 1

        if print_sol:
            logger.info(vpaths) # TODO note that it should output 1-indexed
        return obj

# TODO also do the OVRP from Eduardo's paper



