#!/usr/bin/python3

import argparse
from solver import Solver

def add_parser_args(parser):
    parser.add_argument('--de_pop_size', type=int, default=50, help='DE population size')
    # TODO...

class de(Solver):
    def __init__(self, problem, args):
        super().__init__(problem, args)
        # Preprocessing, etc.
        pass

    def solve(self):
        pass
#         self.problem.evaluate([1,1,1])


