# !/usr/bin/env python
from __future__ import print_function
import math
from utils import *
from operator import mul

class Model:
    def __init__(self, bitext):
        self.bitext = bitext
        self.french_set = set()
        self.english_set = set()
        for (f, e) in self.bitext:
            f.append(None)
            self.french_set.update(f)
            self.english_set.update(e)
        self.fe_set = {(f, e) for f in self.french_set for e in self.english_set}
        self.t = dict.fromkeys(self.fe_set, 1.0 / len(self.english_set))

    def entropy_loss(self):
        res = 0
        for (f, e) in self.bitext:
            prob = reduce(mul, [sum(self.t[(f_i, e_j)] for f_i in f) for e_j in e])
            res -= math.log(prob / len(f) ** len(e))
        return res

    def get_alignment(self):
        for (f, e) in self.bitext:
            for (j, e_j) in enumerate(e):
                best_prob = 0
                best_i = 0
                for (i, f_i) in enumerate(f):
                    if self.t[(f_i, e_j)] > best_prob:
                        best_prob = self.t[(f_i, e_j)]
                        best_i = i
                print("{:d}-{:d} ".format(best_i, j), end="")
            print()

class IBM1(Model):
    opts = {"display":10,"MAX_ITERATION":100,"criteria":1e-5}
    def training(self):
        print("Training with IBM Model 1")
        iter = 0
        loss = 0
        while iter <= self.opts["MAX_ITERATION"]:
            if iter % self.opts["display"] == 0:
                new_loss = self.entropy_loss()
                if abs(loss - new_loss) <= self.opts["criteria"]:
                    print("EM converged at iteration {:04d}\nperplexity: {:.2f}".format(iter, new_loss))
                    break
                print("{:04d} iterations completed\nperplexity: {:.2f}".format(iter, new_loss))
                loss = new_loss

            count = dict.fromkeys(self.fe_set, 0.0)
            total = dict.fromkeys(self.french_set, 0.0)
            for (n, (f_sent, e_sent)) in enumerate(self.bitext):
                # normalization factor
                z = dict.fromkeys(self.english_set, 0)
                for e in e_sent:
                    z[e] = sum(self.t[(f, e)] for f in f_sent)

                for e in e_sent:
                    for f in f_sent:
                        count[(f, e)] += self.t[(f, e)] / z[e]
                        total[f] += self.t[(f, e)] / z[e]
                        # M-step
            for (f, e) in self.fe_set:
                self.t[(f, e)] = count[(f, e)] / total[f]
            iter += 1

class Bayesian(Model):

if __name__ == "__main__":
    f_data, e_data, n = open_file()
    # bitext[][0] for french bitext [][1] for english and split on whitespace
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:n]]

    # bitext = [[["das", "haus"], ["the", "house"]], [["das", "buch"], ["the", "book"]], [["ein", "buch"], ["a", "book"]]]
    ibm = IBM1(bitext)
    ibm.training()
    ibm.get_alignment()
