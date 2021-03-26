#!/usr/bin/python3

import sys
import random

def add_seeds(ff):
    lines = []
    with open(ff, 'r') as fp:
        for line in fp:
            lines.append(f"{random.randint(1, 99999)} {line.strip()}")
    with open(ff, 'w') as fp:
        for line in lines:
            print(line, file=fp)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 add_seeds.py <instance file>")
        sys.exit(1)
    add_seeds(sys.argv[1])

