#!/usr/bin/python3


import argparse
import numba as nb
import numpy.random as npr
import time

def min_max_arg(name, min_val=-1e99, max_val=1e99):
    class MinMaxSize(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if values < min_val or values > max_val:
                parser.error(f"{name} out of range ({values} should be in [{min_val}, {max_val}]")
            setattr(namespace, self.dest, values)
    return MinMaxSize


class Solver(object):
    def __init__(self, problem, args):
        self.args = args
        self.problem = problem

        self.wall_start = time.time()

    def solve(self, problem, instance, args):
        pass

    def terminate(self):
        if time.time() - self.wall_start > self.args.max_wall:
            return True
        elif time.process_time() > self.args.max_cpu:
            return True
        elif self.problem.evaluations > self.args.max_evals:
            return True
        return False

    def random_population(self, size):
        return npr.rand(size, self.problem.dimensions)

