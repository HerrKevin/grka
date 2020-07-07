#!/usr/bin/python3

import argparse
from collections import defaultdict
import numpy as np
import pandas as pd

from problem import Problem

def add_parser_args(parser):
    parser.add_argument('--tugs', type=int, default=2, help='Number of tugs to use')
    parser.add_argument('--labeled', action='store_false', default=True, help='Labeled XLSX instance? (new format)')

def read_instance(inst_path, tugs, lbl):
    if inst_path.endswith('.dzn'):
        inst = parse_instance_dzn(inst_path)
    else:
        if tugs is None:
            print("xls/xlsx instance detected. --tugs must be specified!")
            sys.exit(1)
        inst = parse_instance_xlsx(inst_path) if not lbl else parse_instance_xlsx_labeled(inst_path)
        inst.kk = tugs
    return inst

def parse_instance_dzn(path):
    """
    Note that this is not meant for reading general dzn files... just those
    generated with the scripts in this project
    """
    with open(path, 'r') as fp:
        lines = [line.strip() for line in fp]

    inst = roro_instance()
    read_matrix = False
    target = None
    ii = 0
    for line in lines:
        sp = line.split('=')
        if read_matrix:
            if line.startswith("|];"):
                read_matrix = False
                ii = 0
            else:
                vals = parse_matrix_line(sp[0].replace("[","").strip())
                for jj, vv in enumerate(vals):
                    target[ii,jj] = vv
                ii += 1
        elif line.startswith("nship"):
            inst.nship = int(sp[-1][:-1])
        elif line.startswith("nquay"):
            inst.nquay = int(sp[-1][:-1])
        elif line.startswith("k"):
            inst.kk = int(sp[-1][:-1])
        elif line.startswith("PSS"):
            read_matrix = True
            inst.pss = np.zeros((inst.nship, inst.nship))
            target = inst.pss
            vals = parse_matrix_line(sp[1].replace("[","").strip())
            for jj, vv in enumerate(vals):
                target[ii,jj] = vv
            ii += 1
        elif line.startswith("PQQ"):
            read_matrix = True
            inst.pqq = np.zeros((inst.nquay, inst.nquay))
            target = inst.pqq
            vals = parse_matrix_line(sp[1].replace("[","").strip())
            for jj, vv in enumerate(vals):
                target[ii,jj] = vv
            ii += 1
        elif line.startswith("PSQ"):
            read_matrix = True
            inst.psq = np.zeros((inst.nship, inst.nquay))
            target = inst.psq
            vals = parse_matrix_line(sp[1].replace("[","").strip())
            for jj, vv in enumerate(vals):
                target[ii,jj] = vv
            ii += 1
    return inst

def parse_instance_xlsx(path):
    xlf = pd.read_excel(path, sheet_name=None)
    lldf = xlf['loadlist']
    uldf = xlf['unloadlist']
    pdf = xlf['precedence']

    inst = roro_instance()

    inst.nship = lldf['cargo_id'].max()
    inst.nquay = uldf['cargo_id'].max()
    inst.labeled = False

    inst.psq = np.zeros((inst.nship, inst.nquay))
    np.fill_diagonal(inst.psq, 1)
    inst.dpsq = defaultdict(list)
    inst.drpsq = defaultdict(list)
    # TODO Assumes number of trailers on the ship = # on the quay
    for ii in range(inst.nship):
        inst.dpsq[ii].append(ii)
        inst.drpsq[ii].append(ii)

    inst.pss = np.zeros((inst.nship, inst.nship))
    inst.pqq = np.zeros((inst.nquay, inst.nquay))

    inst.dpss = defaultdict(list)
    inst.dpqq = defaultdict(list)
    inst.drpss = defaultdict(list)
    inst.drpqq = defaultdict(list)

    for row in pdf.itertuples():
        rf,ra = row.fore - 1, row.aft - 1
        inst.pqq[rf, ra] = 1
        inst.pss[ra, rf] = 1

        inst.dpss[ra].append(rf)
        inst.drpss[rf].append(ra)
        inst.dpqq[rf].append(ra)
        inst.drpqq[ra].append(rf)

    return inst

def parse_instance_xlsx_labeled(path):
    xlf = pd.read_excel(path, sheet_name=None)
    sdf = xlf['position_sequence']
    ldf = xlf['precedence_loading']
    udf = xlf['precedence_unloading']

    slots = {}
    slotrev = {}
    for row in sdf.itertuples():
        slots[row.position_onboard] = int(row.dis_seq) - 1
        slotrev[int(row.dis_seq) - 1] = row.position_onboard

    inst = roro_instance()
    inst.nship = max(slots.values()) + 1
    inst.nquay = max(slots.values()) + 1

    inst.slots = slots
    inst.slotrev = slotrev
    inst.labeled = True

    inst.psq = np.zeros((inst.nship, inst.nquay))
    np.fill_diagonal(inst.psq, 1)
    inst.dpsq = defaultdict(list)
    inst.drpsq = defaultdict(list)

    for ii in range(inst.nship):
        inst.dpsq[ii].append(ii)
        inst.drpsq[ii].append(ii)

    inst.pss = np.zeros((inst.nship, inst.nship))
    inst.pqq = np.zeros((inst.nquay, inst.nquay))

    inst.dpss = defaultdict(list)
    inst.dpqq = defaultdict(list)
    inst.drpss = defaultdict(list)
    inst.drpqq = defaultdict(list)

    for row in ldf.itertuples():
        suc = slots[row.suc]
        pred = slots[row.pred]
        inst.pss[suc,pred] = 1

        inst.dpss[suc].append(pred)
        inst.drpss[pred].append(suc)

    for row in udf.itertuples():
        suc = slots[row.suc]
        pred = slots[row.pred]
        inst.pqq[suc,pred] = 1

        inst.dpqq[suc].append(pred)
        inst.drpqq[pred].append(suc)

    return inst




class roro_instance(object):
    def __init__(self):
        self.nship = None
        self.nquay = None
        self.kk = None
        self.pss = None
        self.pqq = None
        self.psq = None

        # Precedence mapping ([i] is blocked by [j, k, ...])
        self.dpss = None
        self.dpqq = None
        self.dpsq = None

        # Reverse precedence mapping ([i] blocks [j, j, ...])
        self.drpss = None
        self.drpqq = None
        self.drpsq = None


class roro(Problem):
    def __init__(self, args):
        self.inst = read_instance(args.instance, args.tugs, args.labeled)
        super().__init__(args, self.inst.nship + self.inst.nquay)


    def batch_evaluate(self, keys, threads=1):
        super().batch_evaluate(keys)
        return [1] * len(keys) # TODO


