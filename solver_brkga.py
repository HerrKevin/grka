from loguru import logger
import argparse
import copy
import numpy as np
import numpy.random as npr
import sys
import time

from solver import Solver, min_max_arg


def add_parser_args(parser):
    parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 5), default=500, help='BRKGA population size')
    parser.add_argument('--generations', type=int, action=min_max_arg('Generations', 1), default=1000, help='BRKGA number of generations')
    parser.add_argument('--elite', type=float, action=min_max_arg('Elite percent', 1e-3,1.0), default=0.1, help='BRKGA elite percentage')
    parser.add_argument('--mutants', type=float, action=min_max_arg('Mutant percent', 1e-3,1.0), default=0.1, help='BRKGA mutant percentage')
    parser.add_argument('--bias', type=float, action=min_max_arg('Crossover elite bias', 0.5,1.0), default=0.6, help='BRKGA mutant percentage')
    parser.add_argument('--loop_to_convergence', action='store_true', default=False, help='Ignore the maximum number of generations and only stop when the population has converged')
    parser.add_argument('--convergence_eps', type=float, default=1e-4, help='Stop if the average change in objective function over the last convergence_last generations falls below this epsilon')
    parser.add_argument('--convergence_last', type=float, default=5, help='Stop if the average change in objective function over the last n generations falls below convergence_eps')


class brkga(Solver):
    def __init__(self, problem, args):
        super().__init__(problem, args)
        if args.elite + args.mutants >= 1.0:
            logger.error(f"Elite percentage plus mutant percentage must be less than 1! ({args.elite} + {args.mutants} = {args.elite + args.mutants}")
            sys.exit(4)

    def terminate(self):
        return super().terminate()

    def solve(self, problem, args):
        pop = self.random_population(self.args.pop_size)
        fitness = self.problem.batch_evaluate(pop, self.args.threads)

        nelite = max(1, int(args.elite * args.pop_size))
        nmutants = max(1, int(args.mutants * args.pop_size))
        nnonelite = args.pop_size - nelite
        recomb = args.pop_size - nelite - nmutants

        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()
        best_val = fitness[best_idx]
        self.status_new_best(best_val, "initial population")

        use_gens = args.generations + 1
        if args.loop_to_convergence:
            use_gens = 2147483647
            last_avgs = [np.mean(fitness), 1e99]
            conv_val = np.abs(np.mean(np.diff(last_avgs)))

        for gg in range(1, use_gens):
            sort_idx = np.argsort(fitness)
            nelite_slice = sort_idx[:nelite]
            elite = pop[nelite_slice,:]
            nonelite = pop[sort_idx[nelite:],:]

            relite_idx = npr.choice(nelite, recomb, replace=True)
            rnonelite_idx = npr.choice(nnonelite, recomb, replace=True)

            recomb_elite = elite[relite_idx,:]
            recomb_nonelite = nonelite[rnonelite_idx,:]

            rm = npr.rand(recomb, self.problem.dimensions)
            recombined = np.where(rm <= args.bias, recomb_elite, recomb_nonelite)

            mutants = self.random_population(nmutants)

            # don't recompute fitness for elite population
            recomb_mutant = np.vstack((recombined, mutants))
            fitness_rm = self.problem.batch_evaluate(recomb_mutant, self.args.threads)
            fitness = np.hstack((fitness[nelite_slice], fitness_rm))
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_val:
                best = pop[best_idx].copy()
                best_val = fitness[best_idx]
                self.status_new_best(best_val, msg=f"Generation {gg}")

            pop = np.vstack((elite, recomb_mutant))

            if args.loop_to_convergence:
                last_avgs.insert(0, np.mean(fitness))
                if len(last_avgs) > args.convergence_last:
                    del last_avgs[-1]
                conv_val = np.abs(np.mean(np.diff(last_avgs)))
                if conv_val < args.convergence_eps:
                    break
            if self.terminate():
                break
        if args.loop_to_convergence and conv_val < args.convergence_eps:
            logger.info(f"Exiting due to convergence criteria ({conv_val} < {args.convergence_eps}; {gg} generations)")

        return best_val, best

