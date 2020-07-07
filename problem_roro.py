#!/usr/bin/python3

import argparse
from problem import Problem

def add_parser_args(parser):
    parser.add_argument('--roro_tugs', type=int, default=2, help='Number of tugs to use')

class roro_instance(object):
    pass

class roro(Problem):
    def __init__(self, args):
        super().__init__(args, 10) # TODO # containers
        # Preprocessing, etc.
        pass

    def batch_evaluate(self, keys, threads=1):
        super().batch_evaluate(keys)
        return [1] * len(keys) # TODO

    def read_instance(self, inst_path):
        pass # TODO
