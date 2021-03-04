#!/usr/bin/python3

from loguru import logger
import argparse
import copy
import numpy as np
import numpy.random as npr
import sys
import time
import importlib as il

from solver import Solver, min_max_arg

import pyade

def add_parser_args(parser):

    parser.add_argument('--ade_alg', choices=['de','ilshade','jade','jso','lshade','lshadecnepsin','mpede','sade','saepsdemmts','shade'], default='ilshade', help='Choice of DE algorithm from pyade')
    parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 50), default=500, help='DE population size')
#     parser.add_argument('--CR', type=float, action=min_max_arg('CR', 0, 1), default=0.3, help='DE parameter CR')
#     parser.add_argument('--F', type=float, action=min_max_arg('F', 0, 2), default=0.8, help='DE parameter F')
#     parser.add_argument('--no_batch_mode', action='store_true', default=False, help='Batch mode performs fitness evaluations in batches instead of sequentially. When this is enabled, batch mode is disabled and things will be slower. Note that batchmode does not use multiple threads; it just vectorizes DE, but that slightly changes the way the algorithm works.')
#     parser.add_argument('--convergence_eps', type=float, default=1e-4, help='Stop if the average change in objective function over the last convergence_last generations falls below this epsilon')
#     parser.add_argument('--convergence_last', type=float, default=5, help='Stop if the average change in objective function over the last n generations falls below convergence_eps')
#     parser.add_argument('--convergence_imp', action='store_true', default=False, help='Stop if the average change in objective function over the last convergence_last generations falls below convergence_eps')


class ade(Solver):
    """
    Wrapper for pyade solvers. Note: no batch mode available
    """
    def __init__(self, problem, args):
        super().__init__(problem, args)
        self.iteration = 0
        self.best_fit = 1e99

    def terminate(self):
        return super().terminate()

    def solve(self):
        solver_ = il.import_module(f'pyade.{self.args.ade_alg}')
        params = solver_.get_default_params(dim=self.problem.dimensions)
        params['population_size'] = self.args.pop_size
        params['max_evals'] = self.args.max_evals
        params['func'] = lambda xx: (self.problem.batch_evaluate(np.array([xx])))[0]
#         params['func'] = lambda xx: np.sum(np.power(xx, 2) - np.cos(xx * 2 * np.pi)) + (10 * self.problem.dimensions)
        params['bounds'] = np.array([[0,1]] * self.problem.dimensions)

        self.best_fit = 1e99

        params['callback'] = lambda **vv: self.callback(**vv)
        params['terminate_callback'] = self.terminate
        sol, fit = solver_.apply(**params)

        return fit, sol

    def callback(self, **vv):
        # assume that pyade keeps track of the best fitness so we don't have to..
        minfit = np.min(vv['fitness'])
        if minfit < self.best_fit:
            self.best_fit = minfit
            self.status_new_best(self.best_fit)






