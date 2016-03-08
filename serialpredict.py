"""
"""
from __future__ import division, print_function
from extractwords import ExtractWords, getPaths
from emaildata import EmailData
from random import shuffle
from sklearn.svm import SVC
from id3 import Id3


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
    sp = SerialPredict(s[:3000], h[:3000])
    vectors = [x + [sp.outputs[i]] for i, x in enumerate(sp.inputs)]
    dec = Id3(vectors[:5000], header=False, test_set=vectors[5000:])
    dec.learnModel()
    right = dec.testExamples(dec.test_set)
    print(right)


def forestTest():
    """
    predict using many decision trees, and voting
    99.8% accuracy on 5000/3000 train test split
    """
    forest = []
    h, s = getPaths("spam", "ham")
    exnum = 5000
    testnum = 3000
    sp = SerialPredict(s[:exnum//2], h[:exnum//2])
    vectors = [x + [sp.outputs[i]] for i, x in enumerate(sp.inputs)]
    numtrees = 25
    for i in range(numtrees):
        print("Learning tree {}.".format(i+1))
        splitnum1 = int((i/numtrees) * testnum)
        splitnum2 = int((i+1/numtrees) * testnum)
        dec = Id3(vectors[splitnum1:splitnum2], header=False)
        dec.titles = sp.vector
        dec.learnModel()
        dec.writeModel("forest/tree{}.txt".format(i))
        forest.append(dec)
    right = 0
    for i, example in enumerate(vectors[testnum:]):
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
    print("Decision forest of {} trees got {} right.".format(numtrees, right/len(vectors[testnum:])))

if __name__ == '__main__':
    # svmTest()
    # id3test()
    forestTest()
