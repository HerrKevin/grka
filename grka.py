#!/usr/bin/python3

import sys
import os.path
import argparse
import importlib
import traceback

def grka(problem, solver, args):
    return solver.solve()


def main():
    parser = argparse.ArgumentParser(description='Generalized Random-Key Algorithm')
    parser.add_argument('problem', choices=['roro'], help='Problem type to solve')
    parser.add_argument('solver', choices=['de'], help='Solver to use to solve the problem')
    parser.add_argument('instance', type=str, help='Instance path')

    args, unparsed_args = parser.parse_known_args(sys.argv[1:])

    if not os.path.exists(args.instance):
        print(f"Instance path does not exist: {args.instance}")
        sys.exit(1)

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
    solver_ = getattr(solver_mod, args.solver)
    problem = prob_(args)
    solver = solver_(problem, args)

    grka(problem, solver, args)

    print(f"Total evaluations: {problem.evaluations}")


if __name__ == "__main__":
    main()


