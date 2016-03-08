"""
emaildata.py
"""
from __future__ import print_function, division
import email                                     # email parser
from datetime import datetime                    # helps for working with dates and times
from bs4 import BeautifulSoup                    # an html parser
import sys
import os
import warnings

__author__ = "Brian Perrett"


class EmailData():

    def __init__(self, emailpath):
        """
        emailpath -> full file path to email
        """
        self.emailpath = emailpath
        self.em = self.getEmailObject()
        self.subject = self.getSubject()
        self.contenttype = self.getContentType()
        self.message = self.getMessage()
        self.sender = self.getSender()
        self.recipient = self.getRecipient()
        self.date = self.getDate()

    def getEmailObject(self):
        """
        reads the file at self.emailpath, and returns a python email object
            from which we can extract data using its get method.
        """
        with open(self.emailpath, "r") as f:
            emobj = email.message_from_string(f.read())
            return emobj

    def getSubject(self):
        try:
            subject = self.em.get("subject")  # get the subject string
            return subject                    # return it if it exists
        except Exception as e:                # if it doesn't, raise exception
            raise(e)

    def getMessage(self):
        try:
            if self.em.is_multipart():                                 # if the msg is multiple parts, put them together
                full_msg = ""
                for payload in self.em.get_payload():
                    if isinstance(payload.get_payload(), basestring):
                        full_msg += payload.get_payload()
            else:                                                      # otherwise, just return the single part message
                full_msg = self.em.get_payload()
        except Exception as e:                                         # if msg doesn't exist, raise exception
            raise(e)
        # parse the text out of html emails.
        if "<html>" in full_msg.lower() or (self.contenttype is not None and "html" in self.contenttype.lower()):
            with warnings.catch_warnings():  # BeautifulSoup gives us a lot of warnings that we don't need to see
                warnings.simplefilter("ignore")
                soup = BeautifulSoup(full_msg, "lxml", fromEncoding="utf-8")
                full_msg = soup.get_text()
                return full_msg
        return full_msg

    def getSender(self):
        try:
            sender = self.em.get("from")  # get the sender string
            return sender
        except Exception as e:            # if msg doesn't exist, raise exception
            raise(e)

    def getRecipient(self):
        try:
            recipient = self.em.get("to")  # get the recipient string
            return recipient
        except Exception as e:             # if msg doesn't exist, raise exception
            raise(e)

    def getContentType(self):
        try:
            contenttype = self.em.get("content-type")  # get contenttype string
            return contenttype
        except Exception as e:                         # if contenttype doesn't exist, raise exception
            raise(e)

    def getDate(self):
        try:
            datestr = self.em.get("date")
            if datestr is None:
                return None
            date = datetime.strptime(" ".join(datestr.split()[:5]), "%a, %d %b %Y %H:%M:%S")
            return date         # return the python datetime object
        except ValueError:
            return None
        except Exception as e:  # if there is an exception, raise it.
            raise(e)


def main():
    emd = EmailData("F:/Box Sync/Documents/Python Documents/APL/Beowulf Cluster/email classification/spam/BG/2004/08/1091394468.23940_19.txt")
    print(emd.subject)
    print(emd.message)
    print(emd.sender)
    print(emd.recipient)
    print(emd.date)
    print("content type: {}".format(emd.contenttype))


if __name__ == '__main__':
    main()
