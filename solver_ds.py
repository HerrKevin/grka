#!/usr/bin/python3

from loguru import logger
import argparse
import copy
import numpy as np
import numpy.random as npr
import sys
import time
import scipy.optimize as spopt

from solver import Solver, min_max_arg

def add_parser_args(parser):
#     parser.add_argument('--min_greedy_interval', type=float, action=min_max_arg('Min greedy gradient interval', 1e-99, 1), default=1e-4, help='Minimum greedy thesis gradient interval')
    parser.add_argument('--min_greedy_subset', type=int, action=min_max_arg('Minimum greedy subset size', 1), default=1, help='Minimum greedy subset size')
    parser.add_argument('--greedy_subset_size', type=float, action=min_max_arg('Greedy subset size', 1e-99, 1), default=0.3, help='Greedy subset size as a percentage of the number of dimensions')

    # TODO parameters of the gradient descent methods
    parser.add_argument('--greedy_gd_method', choices=['Nelder-Mead','BFGS','Powell','CG','L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr'], default='L-BFGS-B', help='Gradient descent technique (see scikit minimize function)')

    parser.add_argument('--antithesis_size', type=float, action=min_max_arg('Antithesis size', 1e-99, 1), default=0.3, help='Perturbation size for antithesis construction')
    parser.add_argument('--no_antithesis_greedy', action='store_true', default=False, help='No greedy improvement of the antithesis')
    parser.add_argument('--min_antithesis_subset', type=int, action=min_max_arg('Minimum antithesis subset size', 1), default=2, help='Minimum antithesis subset size')

    parser.add_argument('--synthesis_method', choices=['path_relinking','gradient_descent'], default='path_relinking', help='Method to use for synthesis')

    parser.add_argument('--synthesis_tol', type=float, action=min_max_arg('Synthesis tolerance', 1e-99, 1e-2), default=1e-4, help='Tolerance for checking thesis/antithesis value equality')

class ds(Solver):
    def __init__(self, problem, args):
        super().__init__(problem, args)
        self.iteration = 0

    def terminate(self):
        return super().terminate()


    # TODO this leaves lots of dimensions essentially as plateaus, which is
    # computationally inefficient for some methods; consider revising so the
    # gradient descent doesn't see the dimensions it's not allowed to change
    def make_mask(self, sol, nn, ro, lbound=None, ubound=None):
        def obj_mask(key):
            keyc = np.copy(sol)
            ro_slice = ro[:nn]
            keyc[ro_slice] = key[ro_slice]
            if lbound is not None:
                keyc = np.maximum(lbound, keyc)
                keyc = np.minimum(ubound, keyc)
            return self.problem.batch_evaluate(np.array([keyc]))[0]
        return obj_mask


    def solve(self):

        # Initialization
        sol = npr.rand(self.problem.dimensions)
        obj = self.problem.batch_evaluate(np.array([sol]))[0]

        ngreedy = int(max(self.args.min_greedy_subset, self.args.greedy_subset_size * self.problem.dimensions))
        nanti = int(max(self.args.min_antithesis_subset, self.args.antithesis_size * self.problem.dimensions))

        rand_order = npr.permutation(self.problem.dimensions)
        itr = 0

        while not self.terminate():

            # Thesis; greedy improvement
            npr.shuffle(rand_order)

            res = spopt.minimize(self.make_mask(sol, ngreedy, rand_order), sol, method=self.args.greedy_gd_method, options={'disp': False})
            if res.success and res.fun < obj:
                obj = res.fun
                sol = np.copy(res.x)
                self.status_new_best(obj, f"iteration {self.iteration}; thesis")

            # Antithesis
            npr.shuffle(rand_order)
            rand_vals = npr.uniform(0, 1, nanti)
            anti = np.copy(sol)
            anti[rand_order[:nanti]] = rand_vals

            if not self.args.no_antithesis_greedy:
                npr.shuffle(rand_order)
                # TODO could allow for a different gradient descent / parameters
                # here... not sure if it is really worth it though
                res = spopt.minimize(self.make_mask(anti, nanti, rand_order), anti, method=self.args.greedy_gd_method, options={'disp': False})
                if res.success and res.fun < obj:
                    obj = res.fun
                    sol = np.copy(res.x)
                    self.status_new_best(obj, f"iteration {self.iteration}; antithesis")

            # Synthesis

            if self.args.synthesis_method == 'path_relinking':
                npr.shuffle(rand_order)
                # Perform path relinking by switching values from the solution to
                # the antithesis in the order given by rand_order
                # TODO this could be performed multiple times?
                sol_synth = np.copy(sol)
                for jj, ro in enumerate(rand_order):
                    if sol_synth[ro] != anti[ro]:
                        sol_synth[ro] = anti[ro]
                        obj_synth = self.problem.batch_evaluate(np.array([sol_synth]))[0]
                        if obj_synth < obj:
                            obj = obj_synth
                            sol = np.copy(sol_synth)
                            self.status_new_best(obj, f"iteration {self.iteration}; synthesis step {jj}/{self.problem.dimensions}")
            else:
                search_idx = []
                lbound = np.zeros(self.problem.dimensions)
                ubound = np.ones(self.problem.dimensions)
                for ii, (tt,aa) in enumerate(zip(sol, anti)):
                    if abs(tt - aa) > self.args.synthesis_tol:
                        search_idx.append(ii)
                        lbound[ii] = min(tt, aa)
                        ubound[ii] = max(tt, aa)
                # Start search at a random point in the convex combination of
                # the thesis and antithesis
                alpha = npr.uniform(0.0, 1.0)
                start_point = alpha * sol + (1 - alpha) * anti
                res = spopt.minimize(self.make_mask(start_point, len(search_idx), search_idx, lbound, ubound), start_point, method=self.args.greedy_gd_method, options={'disp': False})
                if res.success and res.fun < obj:
                    obj = res.fun
                    sol = np.copy(res.x)
            itr += 1
        return obj, sol



