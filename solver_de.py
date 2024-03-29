#!/usr/bin/python3

from loguru import logger
import argparse
import copy
import numpy as np
import numpy.random as npr
import sys
import time

from solver import Solver, min_max_arg

def add_parser_args(parser):

    parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 5), default=50, help='DE population size')
    parser.add_argument('--CR', type=float, action=min_max_arg('CR', 0, 1), default=0.3, help='DE parameter CR')
    parser.add_argument('--F', type=float, action=min_max_arg('F', 0, 2), default=0.8, help='DE parameter F')
    parser.add_argument('--no_batch_mode', action='store_true', default=False, help='Batch mode performs fitness evaluations in batches instead of sequentially. When this is enabled, batch mode is disabled and things will be slower. Note that batchmode does not use multiple threads; it just vectorizes DE, but that slightly changes the way the algorithm works.')
    parser.add_argument('--convergence_eps', type=float, default=1e-4, help='Stop if the average change in objective function over the last convergence_last generations falls below this epsilon')
    parser.add_argument('--convergence_last', type=float, default=5, help='Stop if the average change in objective function over the last n generations falls below convergence_eps')
    parser.add_argument('--convergence_imp', action='store_true', default=False, help='Stop if the average change in objective function over the last convergence_last generations falls below convergence_eps')


class de(Solver):
    def __init__(self, problem, args):
        super().__init__(problem, args)
        self.iteration = 0

    def terminate(self):
        return super().terminate()

    def solve(self):
        # TODO pydoc
        pop = self.random_population(self.args.pop_size)
        fitness = self.problem.batch_evaluate(pop, self.args.threads)

        if self.args.convergence_imp:
            last_avgs = [np.mean(fitness), 1e99]
            conv_val = np.abs(np.mean(np.diff(last_avgs)))

        # TODO minimize vs maximize; right now minimize
        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()
        best_val = fitness[best_idx]
        self.status_new_best(best_val, "initial population")

        while not self.terminate() and (not self.args.convergence_imp or conv_val > self.args.convergence_eps):
            ypop = pop.copy()
            self.iteration += 1
            for xx in range(len(pop)):
                rand_three_idx = npr.choice(len(pop), 3)
                while xx in rand_three_idx:
                    rand_three_idx = npr.choice(len(pop), 3, replace=False)
                aa,bb,cc = pop[rand_three_idx[0]], pop[rand_three_idx[1]], pop[rand_three_idx[2]]

                rnd_idx = npr.randint(self.problem.dimensions)
                rnums = npr.rand(self.problem.dimensions)
                rn_bool = rnums < self.args.CR
                rn_bool[rnd_idx] = True
                rnums_idx = np.where(rn_bool)[0]

                for rr in rnums_idx:
                    ypop[xx,rr] = aa[rr] + self.args.F * (bb[rr] - cc[rr])
                    ypop[xx,rr] = min(max(ypop[xx,rr], 0.0), 1.0) # Box constraints

                if self.args.no_batch_mode:
                    yfitness = self.problem.batch_evaluate(np.array([ypop[xx]]), 1)[0]
                    if yfitness < fitness[xx]:
                        fitness[xx] = yfitness
                        pop[xx] = ypop[xx]
                        if fitness[xx] < best_val:
                            best_val = fitness[xx]
                            best = pop[xx].copy()
                            self.status_new_best(best_val, f"iteration {self.iteration}")
            ## evaluation step (in batch mode; otherwise it's already done)
            if not self.args.no_batch_mode:
                yfitness = self.problem.batch_evaluate(ypop, self.args.threads)
                repl_idx = np.where(yfitness < fitness)
                fitness[repl_idx] = yfitness[repl_idx]
                pop[repl_idx] = ypop[repl_idx]

                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_val:
                    best_val = fitness[best_idx]
                    best = pop[best_idx].copy()
                    self.status_new_best(best_val)

            ## Update convergence criterion
            if self.args.convergence_imp:
                last_avgs.insert(0, np.mean(fitness))
                if len(last_avgs) > self.args.convergence_last:
                    del last_avgs[-1]
                conv_val = np.abs(np.mean(np.diff(last_avgs)))

        if self.args.convergence_imp and conv_val < self.args.convergence_eps:
            logger.info(f"Exiting due to convergence criteria ({conv_val} < {self.args.convergence_eps})")

        return best_val, best





