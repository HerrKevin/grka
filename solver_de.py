#!/usr/bin/python3

import argparse
from solver import Solver, min_max_arg
import copy
import numpy as np
import numpy.random as npr

def add_parser_args(parser):

    parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 5), default=50, help='DE population size')
    parser.add_argument('--CR', type=float, action=min_max_arg('CR', 0, 1), default=0.5, help='DE parameter CR')
    parser.add_argument('--F', type=float, action=min_max_arg('F', 0, 2), default=0.5, help='DE parameter CR')

class de(Solver):
    def __init__(self, problem, args):
        super().__init__(problem, args)
        self.iterations = 0

    def terminate(self):
        return super().terminate()

    def solve(self):
        # TODO pydoc
        pop = self.random_population(self.args.pop_size)
        best = pop[0]
        best_val = 1e-99 # TODO minimize vs maximize
        while not self.terminate():
            for xx in range(len(pop)):
                rand_three_idx = npr.choice(len(pop), 3)
                while xx in rand_three_idx:
                    rand_three_idx = npr.choice(len(pop), 3)
                aa,bb,cc = pop[rand_three_idx[0]], pop[rand_three_idx[1]], pop[rand_three_idx[2]]

                rnd_idx = npr.randint(self.problem.dimensions)
                rnums = npr.rand(self.problem.dimensions)
                rn_bool = rnums < self.args.CR
                rn_bool[rnd_idx] = True
                rnums_idx = np.where(rn_bool)[0]

                for rr in rnums_idx:
                    pop[xx,rr] = aa[rr] + self.args.F * (bb[rr] - cc[rr])
            ## evaluation step
            fitness = self.problem.batch_evaluate(pop, self.args.threads)
            max_fit_idx = np.argmax(fitness)
            if fitness[max_fit_idx] > best_val:
                best_val = fitness[max_fit_idx]
                best = copy.deepcopy(pop[max_fit_idx])

        return best_val, best





