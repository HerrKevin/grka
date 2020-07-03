#!/usr/bin/python3

import threading


class Problem(object):
    def __init__(self, args):
        self.evaluations = 0
        self.eval_lock = threading.Lock()
        self.args = args

    def evaluate(self, key):
        # Evaluation should proceed without changing the class state (so that
        # batch evaluation is possible)
        self.eval_lock.acquire()
        try:
            self.evaluations += 1
        finally:
            self.eval_lock.release()

