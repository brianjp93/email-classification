"""
"""
from __future__ import division, print_function
import os
import sys
from emaildata import EmailData
from bs4 import BeautifulSoup
import warnings

__author__ = "Brian Perrett"


def getPaths(spamfolder, hamfolder):
    """
    extracts the paths to spam and ham email, relative to the <spamfolder>
        and <hamfolder> paths given.
    returns 2 lists of paths to the ham and spam mail.
    """
    hampaths = []
    spampaths = []
    hams = os.listdir(hamfolder)
    for f in hams:
        more = os.listdir("{}/{}".format(hamfolder, f))
        for emails in more:
            hampaths += ["{}/{}/{}/{}".format(hamfolder, f, emails, email) for email in os.listdir("{}/{}/{}".format(hamfolder, f, emails))]
    spams = os.listdir(spamfolder)
    for f in spams:
        more = os.listdir("{}/{}".format(spamfolder, f))
        for folders in more:
            emailfolders = os.listdir("{}/{}/{}".format(spamfolder, f, folders))
            for efolder in emailfolders:
                spampaths += ["{}/{}/{}/{}/{}".format(spamfolder, f, folders, efolder, email) for email in os.listdir("{}/{}/{}/{}".format(spamfolder, f, folders, efolder))]
    return hampaths, spampaths


class ExtractWords():

    def __init__(self,
            hamfolder,
            spamfolder,
            testham=None,
            testspam=None,
            nummsgfeatures=200,
            msgfeaturesfilename="message.fts",
            subjectfeaturesfilename="subject.fts",
            msgfeatures=None):
        self.hampaths, self.spampaths = getPaths(spamfolder, hamfolder)
        self.spammsgwords = {}
        self.hammsgwords = {}
        self.numham = len(self.hampaths)
        self.numspam = len(self.spampaths)
        self.wordlikeliness = {}  # the words with the highest magnitude in this dictionary will probably be good features.
        self.testham = testham if testham is not None else self.numham
        self.testspam = testspam if testspam is not None else self.numspam
        self.nummsgfeatures = nummsgfeatures  # number of message word features to use on each side of self.wordlikeliness
        self.msgfeaturesfilename = msgfeaturesfilename
        self.subjectfeaturesfilename = subjectfeaturesfilename
        self.msgfeatures = self.getMessageFeatures(msgfeatures) if msgfeatures is not None else []

    @staticmethod
    def getMessageFeatures(msgfeatures):
        """
        msgfeatures is the path to the line separated features file
        """
        features = []
        with open(msgfeatures, "r") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    features.append(line)
        return features

    def countWords(self, spampaths, hampaths):
        """
        adds 1 to our count dictionaries if a word is in an email
        """
        for i, em in enumerate(spampaths):  # count up the words in the spam emails
            if i % 10 == 0:
                print("Working on spam email number {}.".format(i))
            messagewords = set()                                 # only add a word once per email
            edata = EmailData(em)
            for word in edata.message.split():
                if word == "":
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    word = BeautifulSoup(word, "lxml").get_text().lower()
                    messagewords.add(word)
            for word in messagewords:
                if word in self.spammsgwords:
                    self.spammsgwords[word] += 1
                else:
                    self.spammsgwords[word] = 1
        for i, em in enumerate(hampaths):   # count up the words in the ham emails
            if i % 10 == 0:
                print("Working on ham email number {}.".format(i))
            messagewords = set()                                # only add a word once per email
            edata = EmailData(em)
            for word in edata.message.split():
                if word == "":
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    word = BeautifulSoup(word, "lxml").get_text().lower()
                    messagewords.add(word)
            for word in messagewords:
                if word == "":
                    continue
                if word in self.hammsgwords:
                    self.hammsgwords[word] += 1
                else:
                    self.hammsgwords[word] = 1
        for word in self.spammsgwords:  # divide by the number of docs to get percentages
            if word == "":
                continue
            self.spammsgwords[word] = self.spammsgwords[word]/len(spampaths)
        for word in self.hammsgwords:
            if word == "":
                continue
            self.hammsgwords[word] = self.hammsgwords[word]/len(hampaths)
        spamwordset = {x for x in self.spammsgwords.keys()}  # get a set of the words in self.spammsgwords
        hamwordset = {x for x in self.hammsgwords.keys()}    # get a set of the words in self.hammsgwords
        spamnotham = spamwordset - hamwordset  # set of words that show up only in spam email
        hamnotspam = hamwordset - spamwordset  # set of words that show up only in ham email
        hamandspam = spamwordset & hamwordset  # set of words that show up in both spam and ham email [intersection]
        for word in spamnotham:
            if word == "":
                continue
            self.wordlikeliness[word] = self.spammsgwords[word]
        for word in hamnotspam:
            if word == "":
                continue
            self.wordlikeliness[word] = -self.hammsgwords[word]
        for word in hamandspam:
            if word == "":
                continue
            self.wordlikeliness[word] = self.spammsgwords[word] - self.hammsgwords[word]
        ratings = sorted(self.wordlikeliness.items(), key=lambda x: x[1])
        for wordrank in ratings[:self.nummsgfeatures]:
            self.msgfeatures.append(wordrank[0])
        for wordrank in ratings[-self.nummsgfeatures:]:
            self.msgfeatures.append(wordrank[0])
        self.writeFeatures(self.msgfeaturesfilename, self.msgfeatures)

    def writeFeatures(self, filename, features):
        with open(filename, "w") as f:
            for feature in features:
                f.write("{}\n".format(feature))



def main():
    # h, s = getPaths("spam", "ham")
    # print(h)
    # print(s)
    ew = ExtractWords("ham", "spam", nummsgfeatures=200, msgfeatures="message.fts")
    # ew = ExtractWords("ham", "spam", nummsgfeatures=200)
    print(ew.numham)
    print(ew.numspam)
    # ew.countWords(ew.spampaths[:200], ew.hampaths[:200])
    # ratings = sorted(ew.wordlikeliness.items(), key=lambda x: x[1])
    # print("hammy: {}".format(ratings[:100]))
    # print("spammy: {}".format(ratings[-100:]))
    print(ew.msgfeatures)


if __name__ == '__main__':
    main()
