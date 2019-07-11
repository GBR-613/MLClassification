import os
import glob
import random
import gensim
import math
import statistics
import datetime
import nltk
import subprocess
from subprocess import run, PIPE
from nltk.corpus import stopwords
from collections import namedtuple
from Data.plots import *
from Preprocess.utils import ArabicNormalizer
from Utils.utils import get_absolute_path, updateParams, show_time, test_path
import stanfordnlp

LabeledDocument = namedtuple('LabeledDocument', 'lines words labels nlabs qLabs name')
stop_words = set(stopwords.words('arabic'))


class DataLoader:
    def __init__(self, Config, DefConfig, kwargs):
        print ("=== Loading data ===")
        updateParams(Config, DefConfig, kwargs)
        self.Config = Config
        self.DefConfig = DefConfig
        self.exclude_categories = Config["exclude_categories"].split(",")
        self.sz = 0
        self.splitTrain = False
        self.topBound = 0.9
        self.charsTopBound = 0.6

        test_path(Config, "train_data_path", "Wrong path to training set. Data can't be loaded.")
        if Config["test_data_path"]:
            test_path(Config, "test_data_path", "Wrong path to testing set. Data can't be loaded.")
        else:
            self.splitTrain = True
            try:
                self.sz = float(Config["test_data_size"])
            except ValueError:
                self.sz = 0
            if not Config["test_data_path"] and (self.sz <= 0 or self.sz >= 1):
                raise ValueError("Wrong size of testing set. Data can't be loaded.")
        if Config["enable_tokenization"] == "True":
            if Config["language_tokenization"] == "True":
                print("GRISHA use single_doc_lang_tokenization")
                if self.Config["use_java"] == "True":
                    test_path(self.Config, 'single_doc_lang_tokenization_lib_path',
                              "Wrong path to the tagger's jar. Preprocessing can't be done.")
                    lib_path = get_absolute_path(Config, 'single_doc_lang_tokenization_lib_path')
                    command_line = 'java -Xmx2g -jar ' + lib_path + ' "' + self.Config["exclude_positions"] + '"'
                    self.jar = subprocess.Popen(command_line, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, shell=True, encoding="utf-8")
                else:
                    self.nlp_tokenizer = stanfordnlp.Pipeline(lang="ar", processors='tokenize,mwt', use_gpu=True)
            if self.Config["stopwords"] == "True":
                self.stopWords = set(nltk.corpus.stopwords.words('arabic'))
            else:
                self.stopWords = set()
            if self.Config["normalization"] == "True":
                self.normalizer = ArabicNormalizer()
        if Config["load_w2v_model"] == "True":
            if not Config["model_path"] or not os.path.isfile(get_absolute_path(Config, "model_path")):
                raise ValueError("Wrong path to W2V model. Stop.")
            try:
                self.ndim = int(self.Config["vectors_dimension"])
            except ValueError:
                raise ValueError("Wrong size of vectors' dimentions. Stop.")
            self.Config["resources"]["w2v"]["created_model_path"] = get_absolute_path(Config, "model_path")
            self.Config["resources"]["w2v"]["ndim"] = self.ndim
            self.load_w2v_model()
        else:
            self.Config["w2vmodel"] = None

        self.load_data()
        if Config["analysis"] == "True":
            self.analysis()

    def load_data(self):
        if self.Config["enable_tokenization"] == "True":
            print("Start loading and preprocessing of data...")
        else:
            print ("Start loading data...")
        ds = datetime.datetime.now()
        self.Config["predefined_categories"] = self.get_categories(get_absolute_path(self.Config, "train_data_path"))
        train_docs = self.get_data_docs(get_absolute_path(self.Config, "train_data_path"))
        if not self.splitTrain:
            test_docs = self.get_data_docs(get_absolute_path(self.Config, "test_data_path"))
        else:
            ind = int(len(train_docs) * (1 - self.sz))
            random.shuffle(train_docs)
            test_docs = train_docs[ind:]
            train_docs = train_docs[:ind]
        de = datetime.datetime.now()
        self.Config["train_docs"] = random.sample(train_docs, len(train_docs))
        self.Config["test_docs"] = random.sample(test_docs, len(test_docs))
        self.get_max_seq_len()
        self.get_max_chars_length()
        if self.Config["enable_tokenization"] == "True" \
                and self.Config["language_tokenization"] == "True" \
                and self.Config["use_java"] == "True":
            self.jar.stdin.write('!!! STOP !!!\n')
            self.jar.stdin.flush()
        print ("Input data loaded in %s"%(show_time(ds, de)))
        print ("Training set contains %d documents."%(len(self.Config["train_docs"])))
        print ("Testing set contains %d documents."%(len(self.Config["test_docs"])))
        print ("Documents belong to %d categories."%(len(self.Config["predefined_categories"])))

    def get_categories(self, path):
        cats = dict()
        nCats = 0
        os.chdir(path)
        for f in glob.glob("*"):
            if os.path.isdir(f) and not f in self.exclude_categories:
                cats[f] = nCats
                nCats += 1
        return cats

    def get_data_docs(self, path):
        files = dict()
        fInCats = [0] * len(self.Config["predefined_categories"])
        nFiles = 0
        actFiles = 0
        curCategory = 0
        docs = []
        os.chdir(path)
        for f in glob.glob("*"):
            if f in self.exclude_categories:
                continue
            curCategory = self.Config["predefined_categories"][f]
            catPath = path + "/" + f
            os.chdir(catPath)
            for fc in glob.glob("*"):
                actFiles += 1
                if fc not in files:
                    nFiles += 1
                    docCont = ''
                    with open(fc, 'r', encoding='UTF-8') as tc:
                        for line in tc:
                            docCont += line.strip() + " "
                    tc.close()
                    if self.Config["enable_tokenization"] == "True":
                        docCont = self.preprocess(docCont)
                    words = docCont.strip().split()
                    labels = [0] * len(self.Config["predefined_categories"])
                    labels[curCategory] = 1
                    nlabs = [f]
                    files[fc] = LabeledDocument(docCont.strip(), words, labels, nlabs, [1], fc)
                else:
                    files[fc].labels[curCategory] = 1
                    files[fc].nlabs.append(f)
                    files[fc].qLabs[0] += 1
                fInCats[curCategory] += 1
        for k, val in files.items():
            docs.append(val)
        return docs

    def get_max_seq_len(self):
        maxDocLen = max(len(x.words) for x in self.Config["train_docs"])
        maxLen = math.ceil(maxDocLen / 100) * 100 + 100
        input_length_list = []
        for i in range(100, maxLen, 100):
            input_length_list.append(i)
        input_length_dict = {x: 0 for x in input_length_list}
        for i in range(len(self.Config["train_docs"])):
            curLen = len(self.Config["train_docs"][i].words)
            dicLen = maxLen
            for ln in input_length_dict:
                if curLen < ln:
                    dicLen = ln
                    break
            input_length_dict[dicLen] = input_length_dict[dicLen] + 1
        input_length_dict_percentage = {}
        for k, v in input_length_dict.items():
            v = v / len(self.Config["train_docs"])
            input_length_dict_percentage[k] = v
        maxSeqLength = 0
        accumulate_percentage = 0
        for length, percentage in input_length_dict_percentage.items():
            accumulate_percentage += percentage
            if accumulate_percentage > self.topBound:
                maxSeqLength = length
                break
        self.Config["max_doc_len"] = maxDocLen
        self.Config["max_seq_len"] = maxSeqLength

    def get_max_chars_length(self):
        maxDocLen = max(len(x.lines) for x in self.Config["train_docs"])
        maxLen = math.ceil(maxDocLen / 100) * 100 + 100
        input_length_list = []
        for i in range(100, maxLen, 100):
            input_length_list.append(i)
        input_length_dict = {x: 0 for x in input_length_list}
        for i in range(len(self.Config["train_docs"])):
            curLen = len(self.Config["train_docs"][i].lines)
            dicLen = maxLen
            for ln in input_length_dict:
                if curLen < ln:
                    dicLen = ln
                    break
            input_length_dict[dicLen] = input_length_dict[dicLen] + 1
        input_length_dict_percentage = {}
        for k, v in input_length_dict.items():
            v = v / len(self.Config["train_docs"])
            input_length_dict_percentage[k] = v
        maxSeqLength = 0
        accumulate_percentage = 0
        for length, percentage in input_length_dict_percentage.items():
            accumulate_percentage += percentage
            if accumulate_percentage > self.charsTopBound:
                maxSeqLength = length
                break
        self.Config["max_chars_doc_len"] = maxDocLen
        self.Config["max_chars_seq_len"] = min(maxSeqLength, 512)

    def analysis(self):
        maxDocLen = max(len(x.words) for x in self.Config["train_docs"])
        minDocLen = min(len(x.words) for x in self.Config["train_docs"])
        avrgDocLen = round(statistics.mean(len(x.words) for x in self.Config["train_docs"]), 2)
        maxCharsDocLen = max(len(x.lines) for x in self.Config["train_docs"])
        minCharsDocLen = min(len(x.lines) for x in self.Config["train_docs"])
        avrgCharsDocLen = round(statistics.mean(len(x.lines) for x in self.Config["train_docs"]), 2)
        dls, qLabs = self.get_label_sets()
        fInCats1 = self.files_by_category(self.Config["train_docs"], self.Config["predefined_categories"])
        fInCats2 = self.files_by_category(self.Config["test_docs"], self.Config["predefined_categories"])
        print("Length of train documents: maximum: %d, minimum: %d, average: %d" % (
                        maxCharsDocLen, minCharsDocLen, avrgCharsDocLen))
        """
        print("Length of %.1f%% documents from training set is less then %d characters." % (
                    self.charsTopBound * 100, self.Config["max_chars_seq_len"]))
        """
        print("Tokens in train documents: maximum: %d, minimum: %d, average: %d" % (maxDocLen, minDocLen, avrgDocLen))
        print("Length of %.1f%% documents from training set is less then %d tokens." % (
            self.topBound * 100, self.Config["max_seq_len"]))
        if self.Config["show_plots"] == "True":
            showDocsByLength(self.Config);
        print("Documents for training in category : maximum: %d, minimum: %d, avegare: %d" % (
        max(fInCats1), min(fInCats1), round(statistics.mean(fInCats1), 2)))
        print("Documents for testing  in category : maximum: %d, minimum: %d, avegare: %d" % (
        max(fInCats2), min(fInCats2), round(statistics.mean(fInCats2), 2)))
        if self.Config["show_plots"] == "True":
            showDocsByLabs(self.Config)
        print("Training dataset properties:")
        print("  Distinct Label Set: %d" % (dls))
        print("  Proportion of Distinct Label Set: %.4f" % (dls / len(self.Config["train_docs"])))
        print("  Label Cardinality: %.4f" % (qLabs / len(self.Config["train_docs"])))
        print("  Label Density: %.4f" % (
                qLabs / len(self.Config["train_docs"]) / len(self.Config["predefined_categories"])))

    def get_label_sets(self):
        labels = [x[2] for x in self.Config["train_docs"]]
        results = [labels[0]]
        qLabs = 0
        for i in range(len(labels)):
            qLabs += sum(labels[i])
            count = 0
            for j in range(len(results)):
                for k in range(len(self.Config["predefined_categories"])):
                    if labels[i][k] != results[j][k]:
                        count += 1
                        break
            if count == len(results):
                results.append(labels[i])
        return len(results), qLabs

    def files_by_category(self, docs, cats):
        fInCats = [0] * len(cats)
        for i in range(len(docs)):
            for j in range(len(cats)):
                if docs[i].labels[j] == 1:
                    fInCats[j] += 1
        return fInCats

    def load_w2v_model(self):
        print ("Load W2V model...")
        ds = datetime.datetime.now()
        self.Config["w2vmodel"] = gensim.models.KeyedVectors.load_word2vec_format(
            get_absolute_path(self.Config, "model_path"))
        de = datetime.datetime.now()
        print("Load W2V model (%s) in %s" % (get_absolute_path(self.Config, "model_path"), show_time(ds, de)))

    def preprocess(self, text):
        if self.Config["language_tokenization"] == "True":
            if self.Config["use_java"] == "True":
                self.jar.stdin.write(text + '\n')
                self.jar.stdin.flush()
                text = self.jar.stdout.readline().strip()
                words = [w for w in text.strip().split() if w not in self.stopWords]
                words = [w for w in words if w not in self.Config["extrawords"]]
            else:
                doc = self.nlp_tokenizer(text)
                words = []
                for sentence in doc.sentences:
                    for token in sentence.tokens:
                        for word in token.words:
                            new_word = word.text
                            if new_word not in self.stopWords and new_word not in self.Config["extrawords"]:
                                words.append(new_word)

            if self.Config["normalization"] == "True":
                words = [self.normalizer.normalize(w) for w in words]
            text = " ".join(words)
        return text


def compose_tsv(model, type):
    cNames = [''] * len(model.Config["predefined_categories"])
    for k, v in model.Config["predefined_categories"].items():
        cNames[v] = k
#    if type == "train":
#        pretrained_bert_model_path = get_absolute_path(model.Config, "resulting_bert_files_path", opt="/train.tsv")
#        data = model.Config[model.keyTrain]
#    else:
#        pretrained_bert_model_path = get_absolute_path(model.Config, "resulting_bert_files_path", opt="/dev.tsv")
#        data = model.Config[model.keyTest]

    pre_trained_bert_model_path = get_absolute_path(model.Config, "resulting_bert_files_path",
                                                    opt=("/train.tsv" if type == "train" else "/dev.tsv"))
    data = model.Config[model.keyTest]
    target = open(pre_trained_bert_model_path, "w", encoding="utf-8")
    for i in range(len(data)):
        conts = data[i].lines.replace('\r','').replace('\n','.')
        nl = '\n'
        if i == 0:
            nl = ''
        string = nl + ",".join(data[i].nlabs) + "\t" + conts
        target.write(string)
    target.close()
