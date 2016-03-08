"""
perceptron.py
Machine Learning - University of Oregon
"""
from __future__ import division
import sys
import csv
import random
import numpy as np

__author__ = "Brian Perrett"


class Perceptron:

    def __init__(self, training=None, test=None, model=None, halt=10, b=0, features=None):
        """
        training        - training file path.
        test            - test file path.
        model           - model file path.  Where to save the model.
        halt            - number of passes before halting algorithm.
        """
        self.features = features  # Set when getTrainingData is run
        self.w = None  # Set when getTrainingData is run
        self.classname = None
        self.training = self.getTrainingData(training) if training is not None else None
        self.test = self.getTestData(test) if test is not None else None
        self.model = model
        self.halt = halt
        self.b = b

    def getTrainingData(self, training=None):
        """
        if training was not specified when initializing Perceptron, then
            it can be specified as an argument to this method.
        """
        if isinstance(training, list):
            self.classname = self.features[-1]
            self.features = self.features[:-1]
            self.w = {feat: 0 for feat in self.features}
            return training
        data = []
        if training is not None:
            self.training = training
        with open(self.training, "r") as f:
            reader = csv.DictReader(f)
            self.features = reader.fieldnames[:-1]
            self.classname = reader.fieldnames[-1]
            self.w = {feat: 0 for feat in self.features}
            for row in reader:
                data.append(row)
        self.training = data
        return data

    def getActivation(self, sample):
        """
        returns True if this guesses the right value.
        """
        a = 0
        y = float(sample[self.classname])
        if y == 0:
            y = -1
        for x in sample:
            if x != self.classname:
                a += float(self.w[x]) * float(sample[x])
        a += self.b
        if y * a <= 0:
            for x in sample:
                if x != self.classname:
                    self.w[x] = self.w[x] + (y * float(sample[x]))
            self.b += y
            return False
        else:
            return True

    def getTestData(self, test_file):
        """
        """
        if isinstance(test_file, list):
            return test_file
        data = []
        with open(test_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        self.test = data
        return data

    def perceptronTrain(self, training=None):
        """
        """
        loops = 0
        converged = False
        while not converged and loops < self.halt:
            if loops % 5 == 0:
                print("Running loop {}".format(loops))
            converged = True
            random.shuffle(self.training)
            for sample in self.training:
                # print sample
                c = self.getActivation(sample)
                if not c:
                    converged = False
            loops += 1

    def testSample(self, x, x_val, w, cname):
        if x != cname:
            return w * x_val
        else:
            return 0

    def perceptronTest(self, sample):
        a = 0
        for x in sample:
            if x != self.classname:
                a += self.w[x] * float(sample[x])
        # a = sum(list(map(lambda x: self.testSample(x, float(sample[x]), self.w[x], self.classname), sample)))
        a += self.b
        print("Activation: {}".format(a))
        if a >= 0:
            return 1
        else:
            return 0

    def saveModel(self):
        with open(self.model, "w") as f:
            f.write("{}\n".format(self.b))
            for x in self.features:
                f.write("{}{}{}\n".format(x, " "*(25 - len(x)), self.w[x]))

    def testModel(self):
        correct = 0
        for sample in self.test:
            a = self.perceptronTest(sample)
            if a == int(sample[self.classname]):
                correct += 1
        return correct / len(self.test)


def testTraining():
    training = sys.argv[1]
    test = sys.argv[2]
    model = sys.argv[3]
    p = Perceptron(training, test, model)
    p.perceptronTrain()
    correct = p.testModel()
    print("Perceptron got {}% correct out of {}".format(correct*100, len(p.test)))
    p.saveModel()


if __name__ == "__main__":
    testTraining()
