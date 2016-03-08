"""
Brian Perrett
1/22/16
University of Oregon - Machine Learning
"""
from __future__ import division
import sys
import csv
from math import log
try:
    from queue import Queue
except:
    from Queue import Queue
import json
import pprint

__author__ = "Brian Perrett"


class Node:
    """
    A node for a binary tree.
    """

    def __init__(self, value=None, entropy=None, parent=None, l=None, r=None, examples=None, side=None, isroot=False, isleaf=False, start=0):
        """
        examples - the index of the examples available at a given node.
            For example, all examples are available at the root.
        """
        self.parent = parent
        self.l = l
        self.r = r
        self.entropy = entropy
        self.value = value
        self.examples = examples
        self.side = side
        self.isroot = isroot
        self.isleaf = isleaf
        self.start = start

    def flatten(self):
        return {
            "value": self.value,
            "left": self.l.flatten() if self.l is not None else None,
            "right": self.r.flatten() if self.r is not None else None
        }


class Id3:
    """
    Implementation of the ID3 decision tree algorithm.
    """
    def __init__(self, dataset, header=True, test_set=None):
        """
        dataset - a filename which is converted to a 2D list of lists
            containing many instances
        header - denotes whether or not the dataset's first row is a header
            (contains titles)
        model - a filename of where the model to be used is.  Optional.  If a
            filename is not provided for model, you will have to create one using
        """
        data = self.getData(dataset)
        if header:
            self.dataset = data[1:]
            self.titles = {x: data[0][x] for x in range(len(self.dataset[0]) - 1)}
        else:
            self.dataset = data
        if test_set is not None:
            self.test_set = self.getData(test_set)
        self.num_attr = len(self.dataset[0]) - 1
        self.classifications = [d[-1] for d in self.dataset]

    @staticmethod
    def ent(classifications):
        """
        computes the entropy of a dataset, given a list of positive
            and negative classifications.
        """
        if len(classifications) == 0:
            return 0
        p = classifications.count(1)/len(classifications)  # positive
        n = classifications.count(0)/len(classifications)  # negative
        # print(p)
        # print(n)
        if p == 0 or n == 0:
            return 0
        ent = (-p * log(p, 2)) - (n * log(n, 2))
        # print(ent)
        return ent

    def learnModel(self):
        """
        The bulk of the ID3 algorithm.
        """
        if self.classifications.count(1) == len(self.classifications):
            return Node(True, 0)
        elif self.classifications.count(0) == len(self.classifications):
            return Node(False, 0)
        elif len(self.dataset[0]) == 0:
            if self.classifications.count(1)/len(self.classifications) >= .5:
                return Node(True, 0)
            else:
                return Node(False, 0)
        else:
            available = [x for x in range(len(self.classifications))]
            ent = self.ent(self.classifications)
            root = Node(examples=available, entropy=ent)
            best = self.findBestAttr(root)
            root = Node(examples=available, entropy=ent, value=best, isroot=True)
            node_stack = [root]
            nodes_tested = 0
            while node_stack:
                # print("Nodes in tree: {}".format(nodes_tested))
                nodes_tested += 1
                node = node_stack.pop()
                # print(node_stack)
                best = node.value
                # print(best)
                # print(best)
                # node.value = best
                # examples are the indeces of the examples in the dataset
                lchild_examples = []
                rchild_examples = []
                data = [self.dataset[x] for x in node.examples]
                for i, x in enumerate(data):
                    # print(i)
                    if x[best] == 0:
                        lchild_examples.append(node.examples[i])
                    else:
                        rchild_examples.append(node.examples[i])
                    # print("left children: {}\nright children: {}".format(len(lchild_examples), len(rchild_examples)))
                # deal with empty example set
                if len(lchild_examples) == 0:
                    if [self.classifications[x] for x in node.examples].count(1)/len(node.examples) >= .5:
                        lchild = Node(True, 0, isleaf=True, side="left", parent=node)
                        node.l = lchild
                    else:
                        lchild = Node(False, 0, isleaf=True, side="left", parent=node)
                        node.l = lchild
                elif [self.classifications[x] for x in lchild_examples].count(1) == len(lchild_examples):
                    lchild = Node(True, 0, isleaf=True, side="left", parent=node)
                    node.l = lchild
                elif [self.classifications[x] for x in lchild_examples].count(0) == len(lchild_examples):
                    lchild = Node(False, 0, isleaf=True, side="left", parent=node)
                    node.l = lchild
                if len(rchild_examples) == 0:
                    if [self.classifications[x] for x in node.examples].count(1)/len(node.examples) >= .5:
                        rchild = Node(True, 0, isleaf=True, side="right", parent=node)
                        node.r = lchild
                    else:
                        rchild = Node(False, 0, isleaf=True, side="right", parent=node)
                        node.r = rchild
                elif [self.classifications[x] for x in rchild_examples].count(1) == len(rchild_examples):
                    rchild = Node(True, 0, isleaf=True, side="right", parent=node)
                    node.r = rchild
                elif [self.classifications[x] for x in rchild_examples].count(0) == len(rchild_examples):
                    rchild = Node(False, 0, isleaf=True, side="right", parent=node)
                    node.r = rchild
                if node.l is None:
                    c = [self.classifications[x] for x in lchild_examples]
                    node.l = Node(parent=node, entropy=self.ent(c), examples=lchild_examples, side="left")
                    best = self.findBestAttr(node.l)
                    if best is None:
                        node.l.value = True
                        node.l.isleaf = True
                        node.l.side = "left"
                    else:
                        node.l.value = best
                        node_stack.append(node.l)
                        # print(lchild_examples)
                        # print("left best: {}".format(best))
                if node.r is None:
                    c = [self.classifications[x] for x in rchild_examples]
                    node.r = Node(parent=node, entropy=self.ent(c), examples=rchild_examples, side="right")
                    best = self.findBestAttr(node.r)
                    if best is None:
                        node.r.value = True
                        node.r.isleaf = True
                    else:
                        node.r.value = best
                        node_stack.append(node.r)
                        # print("right best: {}".format(best))
                # print(node.r.value)
        self.model = root
        return root

    def findBestAttr(self, node):
        examples = [self.dataset[x] for x in node.examples]
        # print(len(examples))
        top_gain = 0
        best = None
        # print(self.num_attr)
        # print(len(self.dataset[0]))
        for attr in range(self.num_attr):
            g = self.gain(node.entropy, examples, attr)
            if g == "skip":
                continue
            if g > top_gain:
                top_gain = g
                best = attr
        # print("top gain: {}".format(top_gain))
        return best

    def chisquared(self, p, n, dp, dn, p_pos, p_neg, n_pos, n_neg):
        """
        returns chi-sqared statistic

        p - number of positively classified examples in D
        n - number of negatively classified examples in D
        dp - number of examples with positive attribute value
        dn - number of examples with negative attribute vlue
        p_pos - number of positively classified examples with positive attribute value
        p_neg - number of positively classified examples with negative attribute value
        n_pos - number of negatively classified examples with positive attribute value
        n_neg - number of negatively classified examples with negative attribute value
        """
        p_hat1 = (p/(p + n)) * dp
        p_hat2 = (p/(p + n)) * dn
        n_hat1 = (n/(p + n)) * dp
        n_hat2 = (n/(p + n)) * dn
        term1 = ((p_pos - p_hat1)**2)/p_hat1 if p_hat1 != 0 else 0
        term2 = ((n_pos - n_hat1)**2)/n_hat1 if n_hat1 != 0 else 0
        term3 = ((p_neg - p_hat2)**2)/p_hat2 if p_hat2 != 0 else 0
        term4 = ((n_neg - n_hat2)**2)/n_hat2 if n_hat2 != 0 else 0
        chi = term1 + term2 + term3 + term4
        return chi

    def gain(self, entropy, examples, attr):
        pos_examples = []
        pos_c = []
        neg_examples = []
        neg_c = []
        # print(examples)
        all_c = [x[-1] for x in examples]
        for x in examples:
            # print(x)
            if int(x[attr]) == 1:
                pos_examples.append(x)
                pos_c.append(x[-1])
                # print(pos_c)
            else:
                neg_examples.append(x)
                neg_c.append(x[-1])
        # chi = self.chisquared(all_c.count("1"), all_c.count("0"), len(pos_examples), len(neg_examples), pos_c.count("1"), neg_c.count("1"), pos_c.count("0"), neg_c.count("0"))
        # print(chi)
        # if chi > 25:
        #     return "skip"
        # print("Positive Classifications: {}".format(pos_c))
        # print("Negative Classifications: {}".format(neg_c))
        g = entropy - ((len(pos_examples)/len(examples)) * self.ent(pos_c)) - ((len(neg_examples)/len(examples)) * self.ent(neg_c))
        # print(g)
        return g

    def classify(self, example):
        """
        attempts to classify an example using the trained model.
        """
        node = self.model
        while node.value is not False and node.value is not True:
            # print(node.value)
            val = node.value
            # print(val)
            if int(example[val]) == 0:
                node = node.l
            else:
                node = node.r
            # print(node.value)
        # print(node.value)
        return node.value

    def testExamples(self, data):
        examples = [x[:-1] for x in data]
        answer = [x[-1] for x in data]
        correct = 0
        for i, x in enumerate(examples):
            guess = self.classify(x)
            # print("found answer")
            if guess is True and int(answer[i]) == 1 or guess is False and int(answer[i]) == 0:
                correct += 1
        return correct/len(data)

    def writeModel(self, filename):
        with open(filename, "w") as f:
            # f.write("hello\n")
            node = self.model
            spaces = 0
            while True:
                if node.start == 0:
                    # print("running == 0")
                    if not node.isleaf:
                        # print("running not is leaf")
                        f.write("\n{}{} = {} :".format("| "*spaces, self.titles[node.value], node.start))
                        node.start += 1
                        node = node.l
                        # print(node)
                        spaces += 1
                    elif node.isleaf:
                        # print("running is leaf")
                        node.start += 1
                        val = 1 if node.value is True else 0
                        f.write(" {}".format(val))
                        node = node.parent
                        spaces -= 1
                elif node.start == 1:
                    # print("running == 1")
                    if not node.isleaf:
                        f.write("\n{}{} = {} :".format("| "*spaces, self.titles[node.value], node.start))
                        node.start += 1
                        node = node.r
                        # print(node)
                        spaces += 1
                    elif node.isleaf:
                        node.start += 1
                        val = 1 if node.value is True else 0
                        f.write(" {}".format(val))
                        node = node.parent
                        spaces -= 1
                else:
                    node.start += 1
                    node = node.parent
                    if node is None:
                        break
                    spaces -= 1

    def getData(self, filename):
        if isinstance(filename, list):
            return filename
        training_set = []
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i != 0:
                    row = [int(x) for x in row]
                training_set.append(row)
        return training_set


def testClassificationOnTraining():
    train = sys.argv[1]
    dec = Id3(train)
    dec.learnModel()
    right = dec.testExamples(dec.dataset)
    # print(dec.model.value)
    print(right)


def testClassificationOnTestSet():
    train = sys.argv[1]
    test = sys.argv[2]
    write_file = sys.argv[3]
    dec = Id3(train, test_set=test)
    dec.learnModel()
    # dec.writeModel(write_file)
    right = dec.testExamples(dec.test_set)
    print(right/len(dec.test_set))


def testLearnModel():
    train = sys.argv[1]
    dec = Id3(train)
    dec.learnModel()


def testGetTrainingSet():
    train = sys.argv[1]
    dec = Id3(train)
    print(dec.titles)
    print(dec.dataset[:5])


def main():
    train = sys.argv[1]
    test = sys.argv[2]
    model = sys.argv[3]
    dec = Id3(train, test_set=test)
    dec.learnModel()
    dec.writeModel(model)

if __name__ == '__main__':
    main()
    # testGetTrainingSet()
    # testLearnModel()
    # testClassificationOnTraining()
    # testClassificationOnTestSet()