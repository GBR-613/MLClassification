import numpy
import pickle
import datetime
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from Utils.utils import show_time, get_absolute_path, arabic_charset


class DataPreparation:
    def __init__(self, model, addValSet):
        self.model = model
        self.addValSet = addValSet
        self.maxWords = 300000
        self.ndim = int(model.Config["vectors_dimension"])
        if addValSet:
            self.validation_data_size = float(model.Config["validation_data_size"])
            if model.Config["type_of_execution"] != "test":
                print("Validation: %.2f" % (self.validation_data_size))
        else:
            self.validation_data_size = 0
        if model.Config["type_of_execution"] == "crossvalidation":
            self.model.cvDocs = model.Config["train_docs"] + model.Config["test_docs"]
            random.shuffle(self.model.cvDocs)
            self.keyTrain = "cross_validations_train_docs"
            self.keyTest = "cross_validations_test_docs"
        else:
            self.keyTrain = "train_docs"
            self.keyTest = "test_docs"

    def getVectors(self, type):
        if type == "wordVectorsSum":
            self.getWordVectorsSum()
        elif type == "wordVectorsMatrix":
            self.getWordVectorsMatrix()
        elif type == "charVectors":
            self.getCharVectors()
        elif type == "vectorize":
            self.getDataForSklearnClassifiers()
        else:
            pass

    def getWordVectorsSum(self):
        self.nfWords = 0
        self.sdict = dict()
        self.tmpCount = 0
        if self.model.Config["type_of_execution"] != "test":
            ds = datetime.datetime.now()
            self.model.trainArrays = numpy.concatenate([self.getDocsArray(x.words, 'Train')
                                                        for x in self.model.Config[self.keyTrain]])
            self.model.trainLabels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(self.model.Config["predefined_categories"]))
                                                        for x in self.model.Config[self.keyTrain]])

            if self.addValSet:
                ind = int(len(self.model.trainArrays) * (1 - self.validation_data_size))
                self.model.valArrays = self.model.trainArrays[ind:]
                self.model.valLabels = self.model.trainLabels[ind:]
                self.model.trainArrays = self.model.trainArrays[:ind]
                self.model.trainLabels = self.model.trainLabels[:ind]

                de = datetime.datetime.now()
                print("Prepare train and validation data in %s" % (show_time(ds, de)))
            else:
                de = datetime.datetime.now()
                print("Prepare train data in %s" % (show_time(ds, de)))

        self.tmpCount = 0
        ds = datetime.datetime.now()
        self.model.testArrays = numpy.concatenate([self.getDocsArray(x.words, "Test")
                                                   for x in self.model.Config[self.keyTest]])
        self.model.testLabels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(self.model.Config["predefined_categories"]))
                                                   for x in self.model.Config[self.keyTest]])
        if self.model.isCV:
            return
        de = datetime.datetime.now()
        print("Prepare test data in %s" % (show_time(ds, de)))
        print("Unique words in all documents: %d" % (len(self.sdict)))
        print("Words not found in the w2v vocabulary: %d" % (self.nfWords))

    def getDocsArray(self, tokens, dataType):
        self.tmpCount += 1
        if self.tmpCount != 0 and self.tmpCount % 1000 == 0:
            print(dataType + ": prepare ", self.tmpCount, end="\r")
        vec = numpy.zeros(self.ndim).reshape((1, self.ndim))
        count = 0.
        for word in tokens:
            if word not in self.sdict:
                self.sdict[word] = 1
            else:
                self.sdict[word] = self.sdict[word] + 1
        for word in tokens:
            try:
                vec += self.model.w2vModel[word].reshape((1, self.ndim))
                count += 1.
            except KeyError:
                if self.sdict[word] == 1:
                    self.nfWords += 1
                continue
        if count != 0:
            vec /= count
        return vec

    def getWordVectorsMatrix(self):
        tokenizer = None
        ds = datetime.datetime.now()
        if self.model.Config["type_of_execution"] != "test":
            tokenizer = Tokenizer(num_words=self.maxWords)
            trainTexts = []
            for t in self.model.Config[self.keyTrain]:
                trainTexts.append(t.lines)
            tokenizer.fit_on_texts(trainTexts)
            if not self.model.isCV:
                with open(get_absolute_path(self.model.Config, "indexer_path"), 'wb') as handle:
                    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
                if self.model.Config["max_doc_len"] > self.model.Config["max_seq_len"]:
                    print("Most of documents from training set have less then %d tokens. Longer documents will be truncated."%(
                        self.model.Config["max_seq_len"]))
            self.model.trainArrays = pad_sequences(tokenizer.texts_to_sequences(trainTexts),
                                                    maxlen=self.model.Config["max_seq_len"])
            self.model.trainLabels = numpy.concatenate([numpy.array(x.labels).
                            reshape(1, len(self.model.Config["predefined_categories"]))
                                                        for x in self.model.Config[self.keyTrain]])
            if self.addValSet:
                ind = int(len(self.model.trainArrays) * (1 - self.validation_data_size))
                self.model.valArrays = self.model.trainArrays[ind:]
                self.model.valLabels = self.model.trainLabels[ind:]
                self.model.trainArrays = self.model.trainArrays[:ind]
                self.model.trainLabels = self.model.trainLabels[:ind]
        if tokenizer == None:
            with open(get_absolute_path(self.model.Config, "indexer_path"), 'rb') as handle:
                tokenizer = pickle.load(handle)
            handle.close()
        testTexts = []
        for t in self.model.Config[self.keyTest]:
            testTexts.append(t.lines)
        self.model.testArrays = pad_sequences(tokenizer.texts_to_sequences(testTexts),
                                              maxlen=self.model.Config["max_seq_len"])
        self.model.testLabels = numpy.concatenate([numpy.array(x.labels).
                            reshape(1, len(self.model.Config["predefined_categories"]))
                                                   for x in self.model.Config[self.keyTest]])
        embedding_matrix = numpy.zeros((self.maxWords, self.ndim))
        word_index = tokenizer.word_index
        nf = 0
        for word, i in word_index.items():
            if i < self.maxWords:
                try:
                    embedding_vector = self.model.w2vModel[word]
                except KeyError:
                    nf += 1
                    continue
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        self.model.embMatrix =  embedding_matrix
        self.model.maxWords = self.maxWords
        if self.model.isCV:
            return
        de = datetime.datetime.now()
        print('Found %s unique tokens.' % len(tokenizer.word_index))
        print ('Tokens not found in W2V vocabulary: %d'%nf)
        print("All data prepared and embedding matrix built in %s"%(show_time(ds, de)))
        return embedding_matrix, self.maxWords

    def getCharVectors(self):
        ds = datetime.datetime.now()
        """
        if self.model.Config["max_chars_doc_len"] > self.model.Config["max_chars_seq_len"]:
            print(
                "Most of documents from training set have less then %d characters. Longer documents will be truncated." % (
                    self.model.Config["max_chars_seq_len"]))
        """
        if self.model.Config["type_of_execution"] != "test":
            self.model.trainArrays = numpy.concatenate([self.stringToIndexes(" ".join(x.words))
                                            for x in self.model.Config[self.keyTrain]])
            self.model.trainLabels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(self.model.Config["predefined_categories"]))
                                                        for x in self.model.Config[self.keyTrain]])
            if self.addValSet:
                ind = int(len(self.model.trainArrays) * (1 - self.validation_data_size))
                self.model.valArrays = self.model.trainArrays[ind:]
                self.model.valLabels = self.model.trainLabels[ind:]
                self.model.trainArrays = self.model.trainArrays[:ind]
                self.model.trainLabels = self.model.trainLabels[:ind]
        self.model.testArrays = numpy.concatenate([self.stringToIndexes(" ".join(x.words))
                                            for x in self.model.Config[self.keyTest]])
        self.model.testLabels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(self.model.Config["predefined_categories"]))
                                                   for x in self.model.Config[self.keyTest]])
        if self.model.isCV:
            return
        de = datetime.datetime.now()
        print("Prepare all data in %s" % (show_time(ds, de)))

    def stringToIndexes(self, str):
        chDict = arabic_charset()
        str2ind = numpy.zeros(self.model.Config["max_chars_seq_len"], dtype='int64')
        strLen = min(len(str), self.model.Config["max_chars_seq_len"])
        for i in range(1, strLen + 1):
            c = str[-i]
            if c in chDict:
                str2ind[i - 1] = chDict[c]
        return str2ind.reshape(1, self.model.Config["max_chars_seq_len"])

    def getDataForSklearnClassifiers(self):
        mlb = None
        ds = datetime.datetime.now()
        if self.model.Config["type_of_execution"] != "test":
            nmCats = [""] * len(self.model.Config["predefined_categories"])
            for k in list(self.model.Config["predefined_categories"].keys()):
                nmCats[self.model.Config["predefined_categories"][k]] = k
            mlb = MultiLabelBinarizer(classes=nmCats)
            wev = (TfidfVectorizer(ngram_range=(1, 3), max_df=0.50)
                   .fit([x.lines for x in self.model.Config[self.keyTrain]],
                                                                [x.nlabs for x in self.model.Config[self.keyTrain]]))
            self.model.trainArrays = wev.transform([x.lines for x in self.model.Config[self.keyTrain]])
            self.model.trainLabels = mlb.fit_transform([x.nlabs for x in self.model.Config[self.keyTrain]])
            if not self.model.isCV:
                with open(get_absolute_path(self.model.Config, "binarizer_path"), 'wb') as handle:
                    pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
                with open(get_absolute_path(self.model.Config, "vectorizer_path"), 'wb') as handle:
                    pickle.dump(wev, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
        if not mlb:
            with open(get_absolute_path(self.model.Config, "binarizer_path"), 'rb') as handle:
                mlb = pickle.load(handle)
            handle.close()
            with open(get_absolute_path(self.model.Config, "vectorizer_path"), 'rb') as handle:
                wev = pickle.load(handle)
            handle.close()
        self.model.testArrays = wev.transform([x.lines for x in self.model.Config[self.keyTest]])
        self.model.testLabels = mlb.fit_transform([x.nlabs for x in self.model.Config[self.keyTest]])
        de = datetime.datetime.now()
        print("Prepare all data in %s" % (show_time(ds, de)))
