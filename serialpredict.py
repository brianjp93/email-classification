"""
"""
from __future__ import division, print_function
from extractwords import ExtractWords, getPaths
from emaildata import EmailData
from random import shuffle, random
from sklearn.svm import SVC
from id3 import Id3
from math import sqrt


class SerialPredict:

    def __init__(self, spampaths, hampaths, findmsgfts=False, msgfts="message.fts"):
        self.spampaths = spampaths
        self.hampaths = hampaths
        self.msgfts = ExtractWords.getMessageFeatures(msgfts)
        self.vector = self.getFeatureVector()
        self.inputs, self.outputs = self.getIO()
        if findmsgfts:
            self.ew = ExtractWords(self.spampaths, self.hampaths, nummsgfeatures=200)
            self.ew.countWords(self.ew.spampaths, self.ew.hampaths)
            self.msgfts = self.ew.msgfeatures

    def getFeatureVector(self):
        vector = []
        vector += self.msgfts
        return vector

    def generateMessageVector(self, emo):
        msgvector = []
        msg = emo.message
        msglist = msg.split()
        for word in self.msgfts:
            if word in msglist:
                msgvector.append(1)
            else:
                msgvector.append(0)
        return msgvector

    def generateFeatureVector(self, emailpath):
        vector = []
        emo = EmailData(emailpath)
        vector += self.generateMessageVector(emo)
        return vector

    def getIO(self):
        vector = []
        for i, spam in enumerate(self.spampaths):
            if i % 100 == 0:
                print("Generating vector for spam number {}.".format(i))
            svec = self.generateFeatureVector(spam) + [1]
            vector.append(svec)
        for i, ham in enumerate(self.hampaths):
            if i % 100 == 0:
                print("Generating vector for ham number {}.".format(i))
            hvec = self.generateFeatureVector(ham) + [0]
            vector.append(hvec)
        shuffle(vector)
        invec = [x[:-1] for x in vector]
        outvec = [x[-1] for x in vector]
        return invec, outvec


def svmTest():
    """
    Decision based on python sklearn-SVM
    """
    h, s = getPaths("spam", "ham")
    sp = SerialPredict(s[:3000], h[:3000])
    # print(sp.inputs)
    # print(sp.outputs)
    trainsize = 5000
    clf = SVC(kernel="linear")
    clf.fit(sp.inputs[:trainsize], sp.outputs[:trainsize])
    testCorrect(trainsize, clf, sp.inputs, sp.outputs)


def id3test():
    """
    Prediction based on a single decision tree
    About 5% less accurate than linear svm.  Maybe will do better in parallel,
        as a decision forest?
    """
    h, s = getPaths("spam", "ham")
    sp = SerialPredict(s[:6500], h[:6500])
    vectors = [x + [sp.outputs[i]] for i, x in enumerate(sp.inputs)]
    dec = Id3(vectors[:10000], header=False, test_set=vectors[10000:])
    dec.learnModel()
    right = dec.testExamples(dec.test_set)
    print(right)


def forestTest():
    """
    """
    forest = []
    h, s = getPaths("spam", "ham")
    exnum = 10000
    testnum = 3000
    sp = SerialPredict(s[:exnum//2], h[:exnum//2])
    sptest = SerialPredict(s[exnum//2: exnum//2 + testnum//2], h[exnum//2: exnum//2 + testnum//2])
    vectors = [x + [sp.outputs[i]] for i, x in enumerate(sp.inputs)]
    testvectors = [x + [sptest.outputs[i]] for i, x in enumerate(sptest.inputs)]
    numtrees = 25
    for i in range(numtrees):
        print("Learning tree {}.".format(i+1))
        splitnum1 = int((i/numtrees) * exnum)
        splitnum2 = int(((i+1)/numtrees) * exnum)
        dec = Id3(vectors[splitnum1:splitnum2], header=False)
        dec.titles = sp.vector
        dec.learnModel()
        dec.writeModel("forest/tree{}.txt".format(i))
        forest.append(dec)
    right = 0
    for i, example in enumerate(testvectors):
        if i % 100 == 0:
            print("Testing example {}.".format(i))
        answers = [0, 0]
        for tree in forest:
            a = tree.classify(example[:-1])
            if a == 0:
                answers[0] += 1
            else:
                answers[1] += 1
        if answers[0] > answers[1]:
            a = 0
        else:
            a = 1
        if a == example[-1]:
            right += 1
    print("Decision forest of {} trees got {} right.".format(numtrees, right/len(testvectors)))


def randomForest():
    """
    not working
    """
    forest = []
    h, s = getPaths("spam", "ham")
    exnum = 10000
    testnum = 3000
    sp = SerialPredict(s[:exnum//2], h[:exnum//2])
    sptest = SerialPredict(s[exnum//2: exnum//2 + testnum//2], h[exnum//2: exnum//2 + testnum//2])
    vectors = [x + [sp.outputs[i]] for i, x in enumerate(sp.inputs)]
    testvectors = [x + [sptest.outputs[i]] for i, x in enumerate(sptest.inputs)]
    numtrees = 25
    treefeatures = []
    for i in range(numtrees):
        print("Learning tree {}.".format(i+1))
        featurechoices = []  # indeces of sqrt(len(example)) features
        baggedvector = []
        for j in range(int(.7 * len(sp.vector))):
            ind = int(random() * len(sp.vector))  # note that the same number could be generated. This is ok.
            featurechoices.append(ind)
        treefeatures.append(featurechoices)
        for k in range(len(vectors)):
            baggedvector.append([vectors[k][j] for j in featurechoices] + [vectors[k][-1]])
        # print("featurechoices: {}".format(featurechoices))
        # print("baggedvector: {}".format(len(baggedvector)))
        # print("baggedvector[0]: {}".format(len(baggedvector[0])))
        # print(len(baggedvector))
        # print(len(baggedvector[0]))
        dec = Id3(baggedvector, header=False)
        dec.titles = [sp.vector[x] for x in featurechoices]
        dec.learnModel()
        dec.writeModel("forest/tree{}.txt".format(i))
        forest.append(dec)
    right = 0
    for i, example in enumerate(testvectors):
        if i % 100 == 0:
            print("Testing example {}.".format(i))
        answers = [0, 0]
        for k, tree in enumerate(forest):
            # print(treefeatures[k])
            # print(len(example))
            examplemod = [example[j] for j in treefeatures[k]] + [example[-1]]
            a = tree.classify(examplemod[:-1])
            if a == 0:
                answers[0] += 1
            else:
                answers[1] += 1
        if answers[0] > answers[1]:
            a = 0
        else:
            a = 1
        if a == examplemod[-1]:
            right += 1
    print("Decision forest of {} trees got {} right.".format(numtrees, right/len(testvectors)))

if __name__ == '__main__':
    # svmTest()
    # id3test()
    # forestTest()
    randomForest()
