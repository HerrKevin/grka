#!/usr/bin/python3

import argparse
import copy
import numpy as np
import numpy.random as npr

import numba
from numba import njit

from parser_helper import min_max_arg

def add_parser_args(parser):
    parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 5), default=50, help='DE population size')
    parser.add_argument('--CR', type=float, action=min_max_arg('CR', 0, 1), default=0.2, help='DE parameter CR')
    parser.add_argument('--F', type=float, action=min_max_arg('F', 0, 2), default=0.5, help='DE parameter CR')


def solve(problem, terminate, args):
    # TODO peval
    return solve_numba(dims=problem.dimensions, peval=None, terminate=terminate, pop_size=args.pop_size, deCR=args.CR, deF=args.F)

@njit
def solve_numba(dims, peval, terminate, pop_size, deCR, deF):
    pop = npr.rand(pop_size, dims)
    best = pop[0]
    best_val = 1e-99 # TODO minimize vs maximize

    evals = 0

    all_idx = np.arange(pop_size, dtype=np.int32)
    fitness_dummy = np.ones(pop_size, dtype=np.float32)

    while not terminate(evals):
        pop_next = pop.copy()
        for xx in range(len(pop)): # prange?
            r3_idx = npr.choice(all_idx, 3)
            while r3_idx[0] == xx or r3_idx[1] == xx or r3_idx[2] == xx:
                r3_idx = npr.choice(all_idx, 3)
            aa,bb,cc = pop[r3_idx[0]], pop[r3_idx[1]], pop[r3_idx[2]]

            rnd_idx = npr.randint(dims)
            rnums = npr.rand(dims)
            rn_bool = rnums < deCR
            rn_bool[rnd_idx] = True
            rnums_idx = np.where(rn_bool)[0]

            for rr in rnums_idx:
                pop_next[xx,rr] = aa[rr] + deF * (bb[rr] - cc[rr])
        ## evaluation step
#         fitness = self.problem.batch_evaluate(pop, self.args.threads)
        fitness = fitness_dummy
        evals += pop_size

        max_fit_idx = np.argmax(fitness)
        if fitness[max_fit_idx] > best_val:
            best_val = fitness[max_fit_idx]
            best = pop[max_fit_idx].copy()
        pop = pop_next
    return best_val, best, evals


# class de(Solver):
#     def __init__(self, problem, args):
#         super().__init__(problem, args)
#         self.iterations = 0
#
#     def terminate(self):
#         return super().terminate()
#
#     def solve(self):
#         # TODO pydoc
#         pop = self.random_population(self.args.pop_size)
#         best = pop[0]
#         best_val = 1e-99 # TODO minimize vs maximize
#         while not self.terminate():
#             pop_next = copy.deepcopy(pop)
#             for xx in range(len(pop)):
#                 rand_three_idx = npr.choice(len(pop), 3)
#                 while xx in rand_three_idx:
#                     rand_three_idx = npr.choice(len(pop), 3)
#                 aa,bb,cc = pop[rand_three_idx[0]], pop[rand_three_idx[1]], pop[rand_three_idx[2]]
#
#                 rnd_idx = npr.randint(self.problem.dimensions)
#                 rnums = npr.rand(self.problem.dimensions)
#                 rn_bool = rnums < self.args.CR
#                 rn_bool[rnd_idx] = True
#                 rnums_idx = np.where(rn_bool)[0]
#
#                 for rr in rnums_idx:
#                     pop_next[xx,rr] = aa[rr] + self.args.F * (bb[rr] - cc[rr])
#             ## evaluation step
#             fitness = self.problem.batch_evaluate(pop, self.args.threads)
#             max_fit_idx = np.argmax(fitness)
#             if fitness[max_fit_idx] > best_val:
#                 best_val = fitness[max_fit_idx]
#                 best = copy.deepcopy(pop[max_fit_idx])
#             pop = pop_next
#
#         return best_val, best
#
#
#
#
#
#
