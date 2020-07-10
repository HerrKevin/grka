#!/usr/bin/python3

from loguru import logger
import argparse
import numpy.random as npr
import signal
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
        self.caught_signal = False

        def sighandler_lambda(sig, frame):
            logger.critical(f"Signal {sig} received; stopping at next opportunity.")
            return self.signal_handler(sig, frame)
        signal.signal(signal.SIGINT, sighandler_lambda)
        signal.signal(signal.SIGTERM, sighandler_lambda)

    def signal_handler(self, sig, frame):
        self.caught_signal = True

    def solve(self, problem, args):
        pass

    def terminate(self):
        if time.time() - self.wall_start > self.args.max_wall:
            return True
        elif time.process_time() > self.args.max_cpu:
            return True
        elif self.problem.evaluations > self.args.max_evals:
            return True
        elif self.caught_signal:
            return True
        return False

    def random_population(self, size):
        return npr.rand(size, self.problem.dimensions)

    def status_new_best(self, val, msg=''):
        if msg:
            msg = f'; {msg}'
        logger.info(f"(Evals {self.problem.evaluations}; CPU {time.process_time():.2f}) New best objective: {val}{msg}")

