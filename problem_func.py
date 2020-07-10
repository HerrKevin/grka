import argparse
import numpy as np
import pandas as pd
import sys

from problem import Problem

def add_parser_args(parser):
    parser.add_argument('--func', choices=['rastrigin'], default='rastrigin', help='Unconstrained, continuous function to solve')
    parser.add_argument('--dims', type=int, default=10, help='Number of dimensions')

def read_instance(inst_path):
    return None

class func(Problem):
    def __init__(self, args, inst):
        super().__init__(args, args.dims)
        self.func = args.func

    def batch_evaluate(self, keys, threads=1):
        super().batch_evaluate(keys)
        keys = (keys * 8.0) - 4.0 # put in range -4,4
        if self.func == 'rastrigin':
            tmp = np.power(keys, 2) - np.cos(keys * 2 * np.pi)
            return np.sum(tmp, axis=1) + (10 * self.dimensions)


