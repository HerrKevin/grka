import argparse
import numpy as np
import pandas as pd
import sys
import cma.bbobbenchmarks as bn
import warnings

from problem import Problem

def add_parser_args(parser):
    parser.add_argument('--func',  type=int, default=1, help='Function to instantiate. For more information, see cma.bbobbenchmarks')
    parser.add_argument('--finst', type=int, default=1, help='Instance of the function to instantiate. For more information, see cma.bbobbenchmarks')
    parser.add_argument('--dims', type=int, default=10, help='Number of dimensions')

    parser.add_argument('--use_gap', action='store_true', default=False, help='Enable this flag to report the objective function as a gap to the optimal value.')

def read_instance(inst_path):
    return None

class func(Problem):
    def __init__(self, args, inst):
        super().__init__(args, args.dims)
        self.func, self.opt = bn.instantiate(args.func, args.finst)
        warnings.filterwarnings('error')

    def batch_evaluate(self, keys, threads=1, print_sol=False):
        super().batch_evaluate(keys)
        keys = (keys * 8.0) - 4.0 # put in range -4,4
        if print_sol:
            print(keys)
        try:
            res = self.func(keys)
        except Warning as ww:
            # TODO pass current best solution up
            raise ValueException(f"Invalid calculation in objective function: {ww}; keys: {keys}")

        if self.args.use_gap:
            res = np.abs((res - self.opt) / self.opt)
        return res
#         if self.func == 'rastrigin':
#             tmp = np.power(keys, 2) - np.cos(keys * 2 * np.pi)
#             return np.sum(tmp, axis=1) + (10 * self.dimensions)
#         elif self.func == 'griewank':
#             right_term = np.prod(np.cos(keys / np.sqrt(np.arange(1, self.dimensions + 1))), axis=1)
#             mid_term = np.sum(np.power(keys, 2) / 4000.0, axis=1)
#             return mid_term + right_term + 1.0
#         elif self.func == 'ackley':
#             aa = 20 # TODO parameterize?
#             bb = 0.2
#             cc = 2 * np.pi
#
#             onedivn = 1.0/len(keys)
#             exp_l = (-1.0 * aa) * np.exp((-1.0 * bb) * np.sqrt(onedivn * np.sum(np.power(keys, 2), axis=1)))
#             exp_r = np.exp(onedivn * np.sum(np.cos(cc * keys), axis=1))
#             return exp_l - exp_r + aa + np.exp(1)
#         elif self.func == 'alpine':
#             return np.sum(np.abs((keys * np.sin(keys)) + (0.1 * keys)), axis=1)
#         elif self.func == 'periodic':
#             # TODO I don't think this one is right
#             return 1.0 + np.sum(np.power(np.sin(keys), 2), axis=1) - 0.1 * np.exp(-1.0 * np.sum(np.power(keys, 2), axis=1))
#         elif self.func == 'salomon':
#             sqk = np.sum(np.power(keys, 2), axis=1)
#             return 1.0 - np.cos(2.0 * np.pi * np.sqrt(sqk)) + 0.1 * np.sqrt(sqk)




