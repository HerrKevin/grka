#!/usr/bin/python3

from loguru import logger
import argparse
import copy
import numpy as np
import numpy.random as npr
import sys
import time
import importlib as il
import warnings

from solver import Solver, min_max_arg

import pyade

def add_parser_args(parser):

    parser.add_argument('--ade_alg', choices=['de','ilshade','jade','jso','lshade','lshadecnepsin','mpede','sade','saepsdemmts','shade'], default='ilshade', help='Choice of DE algorithm from pyade')
    parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 50), default=500, help='DE population size')
    parser.add_argument('--memory_size', type=int, action=min_max_arg('Mem size', 2), default=6, help='((i)l)shade(cnepsin)/jso memory size')
    parser.add_argument('--de_f', type=float, action=min_max_arg('DE_f', 0.0, 2.0), default=0.5, help='DE f')
    parser.add_argument('--de_cr', type=float, action=min_max_arg('DE_cr', 0.0, 1.0), default=0.9, help='DE f')
    parser.add_argument('--de_cross', choices=['bin','exp'], default=bin, help='DE cross')
    parser.add_argument('--jade_p', type=float, action=min_max_arg('JADE_p', 0.0, 1.0), default=0.0, help='JADE/MPEDE p; note: if set less than 1e-3 the parameter will be set to max(0.05, 3.0/pop_size); default value for mpede suggested to be 0.04')
    parser.add_argument('--jade_c', type=float, action=min_max_arg('JADE_c', 0.0, 1.0), default=0.1, help='JADE/MPEDE c')
    parser.add_argument('--mpede_ng', type=int, action=min_max_arg('MPEDE_ng', 1, 200), default=20, help='MPEDE ng')
    parser.add_argument('--mpede_lambda1', type=float, action=min_max_arg('MPEDE_l1', 0.0, 1.0), default=0.2, help='MPEDE lambda1 (note: will be made into prob dist)')
    parser.add_argument('--mpede_lambda2', type=float, action=min_max_arg('MPEDE_l2', 0.0, 1.0), default=0.2, help='MPEDE lambda2 (note: will be made into prob dist)')
    parser.add_argument('--mpede_lambda3', type=float, action=min_max_arg('MPEDE_l3', 0.0, 1.0), default=0.2, help='MPEDE lambda3 (note: will be made into prob dist)')
    parser.add_argument('--mpede_lambda4', type=float, action=min_max_arg('MPEDE_l4', 0.0, 1.0), default=0.4, help='MPEDE lambda4 (note: will be made into prob dist)')


class ade(Solver):
    """
    Wrapper for pyade solvers. Note: no batch mode available
    """
    def __init__(self, problem, args):
        super().__init__(problem, args)
        self.iteration = 0
        self.best_fit = 1e99
        warnings.filterwarnings('error')

    def terminate(self):
        return super().terminate()

    def solve(self):
        alg = self.args.ade_alg
        solver_ = il.import_module(f'pyade.{alg}')
        params = solver_.get_default_params(dim=self.problem.dimensions)
        params['population_size'] = self.args.pop_size
        params['max_evals'] = self.args.max_evals
        params['func'] = lambda xx: (self.problem.batch_evaluate(np.array([xx])))[0]
#         params['func'] = lambda xx: np.sum(np.power(xx, 2) - np.cos(xx * 2 * np.pi)) + (10 * self.problem.dimensions)
        params['bounds'] = np.array([[0,1]] * self.problem.dimensions)

        if alg in ['ilshade', 'lshade', 'shade', 'jso', 'lshadecnepsin']:
            params['memory_size'] = self.args.memory_size
        elif alg == 'de':
            params['f'] = self.args.de_f
            params['cr'] = self.args.de_cr
            params['cross'] = self.args.cross
        elif alg in ['jade', 'mpede']:
            if self.args.jade_p < 1e-3:
                params['p'] = max(0.05, 3.0 / self.args.pop_size)
            else:
                params['p'] = self.args.jade_p
            params['c'] = self.args.jade_c
            if alg == 'mpede':
                params['ng'] = self.args.mpede_ng
                lambdas = np.array([self.args.mpede_lambda1, self.args.mpede_lambda2, self.args.mpede_lambda3, self.args.mpede_lambda4])
                lambdas = lambdas / np.sum(lambdas)
                params['lambdas'] = lambdas


        self.best_fit = 1e99

        params['callback'] = lambda **vv: self.callback(**vv)
        params['terminate_callback'] = self.terminate
        try:
            sol, fit = solver_.apply(**params)
        except Warning as ww:
            if self.args.tuner:
                print(f"GGA SUCCESS {self.best_fit}")
                sys.exit(1)
            else:
                raise ValueError(f"Warning in pyade: {ww}")

        return fit, sol

    def callback(self, **vv):
        # assume that pyade keeps track of the best fitness so we don't have to..
        minfit = np.min(vv['fitness'])
        if minfit < self.best_fit:
            self.best_fit = minfit
            self.status_new_best(self.best_fit)






