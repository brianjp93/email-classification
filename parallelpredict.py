"""
"""
from __future__ import division, print_function
from serialpredict import SerialPredict
from extractwords import getPaths
from mpi4py import MPI
from id3 import Id3
import time

__author__ = "Brian Perrett"


def main():
    t = 51                              # number of trees to make
    exnum = 5000                        # half of the learning set size
    testnum = 1500                      # half of the test set size
    comm = MPI.COMM_WORLD               # how we communicate between cores
    rank = comm.Get_rank()              # The rank of the core running the code
    size = comm.Get_size() - 1          # The number of compute cores we are using
    if rank == 0:                       # The master core
        h, s = getPaths("spam", "ham")  # function from extractwords
        # print(len(h))                   # 19088 ham emails
        # print(len(s))                   # 32988 spam emails
        tpaths = []                     # paths divided into t lists [(ham, spam)]
        testpaths = [[], []]
        for i in range(t):              # separate paths into <t> different lists
            ham = []
            spam = []
            for pindex in range(i, exnum, t):
                ham.append(h[pindex])
            for pindex in range(i, exnum, t):
                spam.append(s[pindex])
            tpaths.append([ham, spam])
        for i in range(exnum, exnum + testnum):
            testpaths[0].append(h[i])
            testpaths[1].append(s[i])
        sp = SerialPredict(testpaths[1], testpaths[0])
        testdata = [x + [sp.outputs[i]] for i, x in enumerate(sp.inputs)]
        coredata = [[] for i in range(size)]  # initialize empty list of lists of size -> # cores
        counter = 0
        while len(tpaths) != 0:
            ind = counter % size              # use modulo to compute index
            data = tpaths.pop()
            coredata[ind].append(data)
            counter += 1
        # print(coredata)
        for i in range(size):
            core = i + 1
            comm.send(coredata[i], dest=core, tag=1)
        for i in range(size):
            core = i + 1
            comm.send(testdata, dest=core, tag=2)
        coreanswers = []
        for i in range(size):
            core = i + 1
            allanswers = comm.recv(source=core, tag=3)
            coreanswers.append(allanswers)
        answercounts = [0] * testnum * 2
        for allanswers in coreanswers:
            for answers in allanswers:
                for i, classification in enumerate(answers):
                    if classification == 1:
                        answercounts[i] += 1
        right = 0
        for i, count in enumerate(answercounts):
            if count > t//2:
                out = 1
            else:
                out = 0
            if sp.outputs[i] == out:
                right += 1
        print("Classifier got {} right out of {} emails.".format(right, testnum*2))
        print(right/(testnum*2))

    if rank != 0:
        load = comm.recv(source=0, tag=1)
        forest = []
        for i, data in enumerate(load):
            sp = SerialPredict(data[1], data[0])
            vectors = [x + [sp.outputs[j]] for j, x in enumerate(sp.inputs)]
            print("Learning tree {} on core {}.".format(i, rank))
            dec = Id3(vectors, header=False)
            dec.titles = sp.vector
            dec.learnModel()
            forest.append(dec)
        vectors = comm.recv(source=0, tag=2)  # receive the test data
        allanswers = []
        for tree in forest:
            answers = []
            for i, example in enumerate(vectors):
                a = tree.classify(example[:-1])
                answers.append(a)
            allanswers.append(answers)
        comm.send(allanswers, dest=0, tag=3)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Total time: {} seconds.".format(end-start))
