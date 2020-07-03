#!/usr/bin/python3

import argparse
from problem import Problem

def add_parser_args(parser):
    parser.add_argument('--roro_tugs', type=int, default=2, help='Number of tugs to use')

class roro(Problem):
    def __init__(self, args):
        super().__init__(args)
        # Preprocessing, etc.
        pass

    def evaluate(self, key):
        super().evaluate(key)
        pass

