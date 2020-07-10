from loguru import logger
import argparse
import copy
import numpy as np
import numpy.random as npr
import sys
import time

from solver import Solver, min_max_arg

def add_parser_args(parser):
    parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 5), default=50, help='PSO population size')
    parser.add_argument('--phi_p', type=float, action=min_max_arg('Phi P', 0, 4), default=2.05, help='PSO parameter phi P; phi_p + phi_g > 4')
    parser.add_argument('--phi_g', type=float, action=min_max_arg('Phi G', 0, 4), default=2.05, help='PSO parameter phi G; phi_p + phi_g > 4')
    parser.add_argument('--convergence_eps', type=float, default=1e-4, help='Stop if the average change in objective function over the last convergence_last generations falls below this epsilon')
    parser.add_argument('--convergence_last', type=float, default=5, help='Stop if the average change in objective function over the last n generations falls below convergence_eps')
    parser.add_argument('--convergence_imp', action='store_true', default=False, help='Stop if the average change in objective function over the last convergence_last generations falls below convergence_eps')


class pso(Solver):
    def __init__(self, problem, args):
        super().__init__(problem, args)
        self.iteration = 0

    def terminate(self):
        return super().terminate()

    def solve(self):
        if self.args.phi_p + self.args.phi_g <= 4:
            logger.error(f"Phi p + phi g <= 4 ({self.args.phi_p} + {self.args.phi_g} <= 4), which leads to an invalid constriction factor.")
            sys.exit(5)
        phi = self.args.phi_p + self.args.phi_g
        chi = 2.0 / (phi - 2 + np.sqrt((phi * phi) - 4.0 * phi))
        logger.info(f"Constriction factor: {chi}")

        # TODO pydoc
        pos = self.random_population(self.args.pop_size)
        vel = (self.random_population(self.args.pop_size) / 50.0) - 0.01
        bpos = pos.copy()

        fitness = self.problem.batch_evaluate(pos, self.args.threads)
        bpfitness = fitness.copy()

        if self.args.convergence_imp:
            last_avgs = [np.mean(fitness), 1e99]
            conv_val = np.abs(np.mean(np.diff(last_avgs)))

        # TODO minimize vs maximize; right now minimize
        best_idx = np.argmin(fitness)
        best = pos[best_idx].copy()
        best_val = fitness[best_idx]
        self.status_new_best(best_val, "initial population")

        while not self.terminate() and (not self.args.convergence_imp or conv_val > self.args.convergence_eps):
            self.iteration += 1

            # Position update
            rnd_phi_p = npr.uniform(0, self.args.phi_p, (self.args.pop_size, self.problem.dimensions))
            rnd_phi_g = npr.uniform(0, self.args.phi_g, (self.args.pop_size, self.problem.dimensions))
            vel = (vel + ((bpos - pos) * rnd_phi_p) + ((best - pos) * rnd_phi_g)) * chi
            pos = pos + vel

            # Enforce box constraints
            pos = np.minimum(np.maximum(pos, 0.0), 1.0)

            fitness = self.problem.batch_evaluate(pos, self.args.threads)

            # Update best known particle position
            repl_idx = np.where(fitness < bpfitness)
            bpos[repl_idx] = pos[repl_idx]
            bpfitness[repl_idx] = fitness[repl_idx]

            # Update global best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_val:
                best = pos[best_idx].copy()
                best_val = fitness[best_idx]
                self.status_new_best(best_val, f"iteration {self.iteration}")

            ## Update convergence criterion
            if self.args.convergence_imp:
                last_avgs.insert(0, np.mean(fitness))
                if len(last_avgs) > self.args.convergence_last:
                    del last_avgs[-1]
                conv_val = np.abs(np.mean(np.diff(last_avgs)))

        if self.args.convergence_imp and conv_val < self.args.convergence_eps:
            logger.info(f"Exiting due to convergence criteria ({conv_val} < {self.args.convergence_eps})")

        return best_val, best



