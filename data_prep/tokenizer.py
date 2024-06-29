import re
import string

from config import TIMESTAMPED
from joblib import dump
from nltk.tokenize import TweetTokenizer
from data_prep.sclc import SCLConverter
from timestamper import TimeStamper
from utilize_data import UtilizeData


class Tokenizer:
    def __init__(self, data):
        self.data = data
        self.tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        self.actions = {}
        self.sclc_run_all().create_actors().create_sentence().initial_tokenize()
        self.utilizedata = UtilizeData(self.actions)

    def sclc_run_all(self):
        sclc = SCLConverter(self.data)
        self.data = sclc.deepcopied
        if TIMESTAMPED == True:
            timestamper = TimeStamper(self.data)
            self.data = timestamper.deepcopied

        return self

    def create_actors(self):
        deleteactors = []
        for actor in self.data.keys():
            if self.data[actor] == {}:
                deleteactors.append(actor)

        for actor in deleteactors:
            del self.data[actor]

        return self

    def create_sentence(self):
        for actor in self.data.keys():
            sentences = []
            for victim in self.data[actor]:
                if self.data[actor][victim] != []:
                    sentence = ""
                    for i in range(len(self.data[actor][victim])):
                        sentence += self.data[actor][victim][i]["data"].lower()
                        if i != len(self.data[actor][victim]) - 1:
                            sentence += " "
                    sentences.append(sentence)
            self.actions[actor] = sentences

        return self

    def initial_tokenize(self):
        to_be_dropped = []
        for actor in self.actions.keys():
            sentences = []
            for sentence in self.actions[actor]:
                new_sentence = self.tokenizer.tokenize(sentence)
                new_sentence = self.preprocess(new_sentence)
                sentences.append(new_sentence)

            self.actions[actor] = sentences
            if len(sentences) == 0:
                to_be_dropped.append(actor)

        for each in to_be_dropped:
            del self.actions[each]

        return self

    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    def preprocess(self, sentences):
        new_sentences = []
        for x in sentences:
            if re.fullmatch("[" + string.punctuation + "]+", x):
                continue
            if x == "\\":
                continue
            if x == ":\\":
                continue
            if x.isdigit() == True:
                continue
            if self.isfloat(x) == True:
                continue
            x = x.replace("https://", "").replace("http://", "").replace("www", "")
            new_sentences.append(x)

        return new_sentences
