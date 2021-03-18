from loguru import logger
import argparse
import copy
import numpy as np
import numpy.random as npr
import sys
import time
import cma
import warnings
import sys

from solver import Solver, min_max_arg

def add_parser_args(parser):
    # TODO this should really be a subparser for all solvers, then the help
    # command would work correctly

    parser.add_argument('--AdaptSigma',                 choices=['True','False','CMAAdaptSigmaTPA','CMAAdaptSigmaCSA','CMAAdaptSigmaNone','CMAAdaptSigmaDistanceProportional','CMAAdaptSigmaMedianImprovement'], default='True', help='CMA population size')
    parser.add_argument('--CMA_active',                 type=bool, default=True, help='CMA_active')
    parser.add_argument('--CMA_cmean',                  type=float, default=1, help='CMA_cmean')
    parser.add_argument('--CMA_elitist',                choices=['True','False','initial'], default=True, help='CMA_elitist')
    parser.add_argument('--CMA_on',                     type=float, default=1, help='CMA_on')
    parser.add_argument('--CMA_rankmu',                 type=float, default=1.0, help='CMA_rankmu')
    parser.add_argument('--CMA_rankone',                type=float, default=1.0, help='CMA_rankone')
    parser.add_argument('--CSA_dampfac',                type=float, default=1, help='CSA_dampfac')
    parser.add_argument('--CSA_damp_mueff_exponent',    type=float, default=0.5, help='CSA_damp_mueff_exponent')
    parser.add_argument('--CSA_disregard_length',       type=bool, default=False, help='CSA_disregard_length')
    parser.add_argument('--CSA_squared',                type=bool, default=False, help='CSA_squared')
    parser.add_argument('--maxiter',                    type=str, default='100 + 150 * (N+3)**2', help='maxiter')
    parser.add_argument('--mean_shift_line_samples',    type=bool, default=False, help='mean_shift_line_samples')
    parser.add_argument('--tolconditioncov',            type=float, default=1e14, help='tolconditioncov')
    parser.add_argument('--tolfacupx',                  type=float, default=1e3, help='tolfacupx')
    parser.add_argument('--tolflatfitness',             type=int, default=1, help='tolflatfitness')
    parser.add_argument('--tolfun',                     type=float, default=1e-11, help='tolfun')
    parser.add_argument('--tolfunhist',                 type=float, default=1e-12, help='tolfunhist')
    parser.add_argument('--tolfunrel',                  type=int, default=0, help='tolfunrel')
    parser.add_argument('--tolx',                       type=float, default=1e-11, help='tolx')
    parser.add_argument('--CMA_mirrormethod',           type=int, action=min_max_arg('', 0, 2), default=2, help='CMA_mirrormethod')


class cmaes(Solver):
    def __init__(self, problem, args):
        super().__init__(problem, args)
        self.iteration = 0
        warnings.filterwarnings('error')

    def terminate(self):
        return super().terminate()

    def solve(self):
        best_val = 1e99
        topts = ['AdaptSigma', 'CMA_active', 'CMA_cmean', 'CMA_elitist',
                'CMA_on', 'CMA_rankmu', 'CMA_rankone', 'CSA_dampfac',
                'CSA_damp_mueff_exponent', 'CSA_disregard_length',
                'CSA_squared', 'maxiter', 'mean_shift_line_samples',
                'tolconditioncov', 'tolfacupx', 'tolflatfitness', 'tolfun',
                'tolfunhist', 'tolfunrel', 'tolx', 'CMA_mirrormethod']

        opts = cma.CMAOptions()
        for to in topts:
            opts[to] = getattr(self.args, to)
        opts['maxfevals'] = self.args.max_evals
        opts['verbose'] = -1
        if self.args.seed >= 0:
            opts['seed'] = self.args.seed
        opts['bounds'] = [0.0, 1.0]
        # TODO ftarget for changing the sense inf / -inf


        sigma0 = 0.5 / 3.0

        es = cma.CMAEvolutionStrategy([0.5] * self.problem.dimensions, sigma0, opts)

        try:
            # TODO check versitile options for hyperreactive version!

    #         while not es.stop() and not self.terminate():
            while not self.terminate():
                solutions = es.ask()
                answers = self.problem.batch_evaluate(np.array(solutions))
                es.tell(solutions, answers)
                res = es.result
                if res.fbest < best_val:
                    best_val = res.fbest
                    self.status_new_best(best_val)
                #es.disp()
            res = es.result

            return res.fbest, res.xbest
        except Warning as ww:
            if self.args.tuner:
                # An error happened, just return what we have
                return res.fbest, res.xbest
#                 res = es.result
#                 print(f"GGA CRASHED {res.fbest}")
#                 sys.exit(1)
            else:
                raise ValueError(f"Warning raised in CMA-ES: {ww.message}")





