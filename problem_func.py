import argparse
import numpy as np
import pandas as pd
import sys

from problem import Problem

def add_parser_args(parser):
    parser.add_argument('--func', choices=['rastrigin','griewank','ackley'], default='rastrigin', help='Unconstrained, continuous function to solve')
    parser.add_argument('--dims', type=int, default=10, help='Number of dimensions')

def read_instance(inst_path):
    return None

class func(Problem):
    def __init__(self, args, inst):
        super().__init__(args, args.dims)
        self.func = args.func

    def batch_evaluate(self, keys, threads=1, print_sol=False):
        super().batch_evaluate(keys)
        keys = (keys * 8.0) - 4.0 # put in range -4,4
        if print_sol:
            print(keys)
        if self.func == 'rastrigin':
            tmp = np.power(keys, 2) - np.cos(keys * 2 * np.pi)
            return np.sum(tmp, axis=1) + (10 * self.dimensions)
        elif self.func == 'griewank':
            right_term = np.prod(np.cos(keys / np.sqrt(np.arange(1, self.dimensions + 1))), axis=1)
            mid_term = np.sum(np.power(keys, 2) / 4000.0, axis=1)
            return mid_term + right_term + 1.0
        elif self.func == 'ackley':
            aa = 20
            bb = 0.2
            cc = 2 * np.pi

            onedivn = 1.0/len(keys)
            exp_l = (-1.0 * aa) * np.exp((-1.0 * bb) * np.sqrt(onedivn * np.sum(np.power(keys, 2), axis=1)))
            exp_r = np.exp(onedivn * np.sum(np.cos(cc * keys), axis=1))
            return exp_l - exp_r + aa + np.exp(1)




