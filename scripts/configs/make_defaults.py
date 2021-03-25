#!/usr/bin/python3

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python3 make_defaults.py <scenario> <prefix> <template (optional)>")
        sys.exit(1)

    scenario = sys.argv[1]
    prefix = sys.argv[2]
    if len(sys.argv) == 4:
        template = []
        with open(sys.argv[3], 'r') as fp:
            template = [line.strip() for line in fp]
    solvers = {
            'cmaes': "--solver cmaes",
            'brkga': "--solver brkga",
            'pso': "--solver pso",
            'random': "--solver random",
            'de': "--solver ade --ade_alg de",
            'ilshade': "--solver ade --ade_alg ilshade",
            'jade': "--solver ade --ade_alg jade",
            'jso': "--solver ade --ade_alg jso",
            'lshade': "--solver ade --ade_alg lshade",
            'lshadecnepsin': "--solver ade --ade_alg lshadecnepsin",
            'mpede': "--solver ade --ade_alg mpede",
            'sade': "--solver ade --ade_alg sade",
            'shade': "--solver ade --ade_alg shade",
            }
    
    for sstr,pstr in solvers.items():
        with open(f'train_{prefix}_{sstr}_default.sh', 'w') as fp:
            print(f"inst_file=tuning/{scenario}/instances.txt", file=fp) # TODO probably should be moved into template
            print(f"job_name=train_{prefix}_{sstr}_default", file=fp)
            for line in template:
                print(line, file=fp)
            print(f"params=\"{pstr}\"", file=fp)

        with open(f'test_{prefix}_{sstr}_default.sh', 'w') as fp:
            print(f"inst_file=tuning/{scenario}/instances_test.txt", file=fp)
            print(f"job_name=test_{prefix}_{sstr}_default", file=fp)
            for line in template:
                print(line, file=fp)
            print(f"params=\"{pstr}\"", file=fp)


