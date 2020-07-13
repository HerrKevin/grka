from loguru import logger
import argparse
import copy
import numpy as np
import numpy.random as npr
import sys
import time
from scipy.spatial.distance import pdist, squareform

from solver import Solver, min_max_arg

def add_parser_args(parser):
    nfeats = 6
    wdefault = [0]*nfeats

    parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 5), default=50, help='PSO population size')
    parser.add_argument('--phi_p_weights', nargs=nfeats, type=float, default=wdefault, help='PSO Hyper configurable weights for phi p')
    parser.add_argument('--phi_g_weights', nargs=nfeats, type=float, default=wdefault, help='PSO Hyper configurable weights for phi g')
    parser.add_argument('--chi_weights', nargs=nfeats, type=float, default=wdefault, help='PSO Hyper configurable weights for chi')

    parser.add_argument('--max_phi_p', type=int, action=min_max_arg('Max phi P', 1), default=5, help='PSO maximum value of phi p')
    parser.add_argument('--max_phi_g', type=int, action=min_max_arg('Max phi G', 1), default=5, help='PSO maximum value of phi g')

    parser.add_argument('--convergence_eps', type=float, default=1e-4, help='Stop if the average change in objective function over the last convergence_last generations falls below this epsilon')
    parser.add_argument('--convergence_last', type=float, default=5, help='Stop if the average change in objective function over the last n generations falls below convergence_eps')
    parser.add_argument('--convergence_imp', action='store_true', default=False, help='Stop if the average change in objective function over the last convergence_last generations falls below convergence_eps')


class hcpso(Solver):
    def __init__(self, problem, args):
        super().__init__(problem, args)
        self.iteration = 0

    def terminate(self):
        return super().terminate()

    def solve(self):
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

        cpu_last_imp = time.process_time()
        itr_last_imp = 1
        feat_pct_oob = 0.0

        while not self.terminate() and (not self.args.convergence_imp or conv_val > self.args.convergence_eps):
            self.iteration += 1

            ### Compute features; for use in computing parameters

            # compute feature f from APSO paper from Zhan et al.
            dists = squareform(pdist(pos, 'euclidean'))
            avg_dists = np.mean(dists, axis=0)
            dg = avg_dists[best_idx]
            dmin = np.min(avg_dists)
            dmax = np.max(avg_dists)
            feat_f = (dg - dmin) / (dmax - dmin)

            feat_wbb = (np.min(fitness) - best_val) / best_val
            # TODO Even with the logs there's still probably a risk of some kind of
            # overflow here
            feat_cpu_since = np.log(1 + time.process_time() - cpu_last_imp)
            feat_iters_since = np.log(1 + self.iteration - itr_last_imp)
            feat_avg_vel = np.mean(vel)
            feats = np.array([feat_f, feat_wbb, feat_cpu_since, feat_iters_since, feat_pct_oob, feat_avg_vel])

            # Note: the max value of 5 centers phi p and phi g around 2.5 for
            # empty values in the feature vector, meaning it is close to the
            # "really good" values in the literature; probably an implicit bias
            phi_p = 1.0 / (1.0 + np.exp(np.dot(feats, self.args.phi_p_weights))) * self.args.max_phi_p
            phi_g = 1.0 / (1.0 + np.exp(np.dot(feats, self.args.phi_g_weights))) * self.args.max_phi_g
            chi = 1.0 / (1.0 + np.exp(np.dot(feats, self.args.chi_weights)))

            logger.info(f"feats {['%.2f'%ii for ii in feats]}; phi p {phi_p}, phi g {phi_g}, chi {chi}") # TODO output this information to a file?

            # Position update
            rnd_phi_p = npr.uniform(0, phi_p, (self.args.pop_size, self.problem.dimensions))
            rnd_phi_g = npr.uniform(0, phi_g, (self.args.pop_size, self.problem.dimensions))
            vel = (vel + ((bpos - pos) * rnd_phi_p) + ((best - pos) * rnd_phi_g)) * chi
            pos = pos + vel

            # Enforce box constraints
            feat_pct_oob = (np.sum(pos < 0.0) + np.sum(pos > 1.0)) / self.args.pop_size
            pos = np.minimum(np.maximum(pos, 0.0), 1.0)

            fitness = self.problem.batch_evaluate(pos, self.args.threads)

            # Update best known particle position
            repl_idx = np.where(fitness < bpfitness)
            bpos[repl_idx] = pos[repl_idx]
            bpfitness[repl_idx] = fitness[repl_idx]

            # Update global best
            ybest_idx = np.argmin(fitness)
            if fitness[ybest_idx] < best_val:
                best_idx = ybest_idx
                best = pos[best_idx].copy()
                best_val = fitness[best_idx]
                itr_last_imp = self.iteration
                cpu_last_imp = time.process_time()
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



