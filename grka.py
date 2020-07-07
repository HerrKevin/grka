#!/usr/bin/python3

import sys
import os.path
import argparse
import importlib
import traceback
import time
import random
import numpy.random as npr
from terminator import create_terminator

def grka(wall_start, problem, solver, args):
    terminate = create_terminator(wall_start, args)
    return solver.solve(problem, terminate, args)


def main():
    wall_start = time.time()

    parser = argparse.ArgumentParser(description='Generalized Random-Key Algorithm')
    parser.add_argument('problem', choices=['roro'], help='Problem type to solve')
    parser.add_argument('solver', choices=['de'], help='Solver to use to solve the problem')
    parser.add_argument('instance', type=str, help='Instance path')
    parser.add_argument('--max_cpu', type=float, default=1e99, help='Maximum CPU time in seconds')
    parser.add_argument('--max_wall', type=float, default=1e99, help='Maximum wall time in seconds')
    parser.add_argument('--max_evals', type=int, default=int(1e15), help='Maximum number of objective function evaluations')
    parser.add_argument('--threads', type=int, default=1, help='Maximum number of threads to use')
    parser.add_argument('-s', '--seed', type=int, default=-1, help='Random seed (default = -1, meaning no seed given)')

    args, unparsed_args = parser.parse_known_args(sys.argv[1:])

    if not os.path.exists(args.instance):
        print(f"Instance path does not exist: {args.instance}")
        sys.exit(1)

    if args.seed >= 0:
        random.seed(args.seed)
        npr.seed(args.seed)

    prob_mod = f"problem_{args.problem}"
    try:
        problem_mod = importlib.__import__(prob_mod)
    except:
        print(f"Error loading problem module {prob_mod}")
        print(traceback.format_exc())
        sys.exit(2)

    solve_mod = f"solver_{args.solver}"
    try:
        solver_mod = importlib.__import__(solve_mod)
    except:
        print(f"Error loading solver module {solve_mod}")
        print(traceback.format_exc())
        sys.exit(3)

    # reparse arguments now for specific problem and solver

    problem_mod.add_parser_args(parser)
    solver_mod.add_parser_args(parser)

    # TODO use unparsed_args here and merge?
    args = parser.parse_args(sys.argv[1:])

    prob_ = getattr(problem_mod, args.problem)
    problem = prob_(args)

    best_val, best, evals = grka(wall_start, problem, solver_mod, args)

    print(f"Total evaluations: {evals}")


if __name__ == "__main__":
    main()


