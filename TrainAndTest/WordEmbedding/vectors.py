import os
import numpy
import datetime
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from Utils.utils import get_abs_path, get_formatted_date, updateParams, test_path


class Embedding:
    def __init__(self, Config, DefConfig, kwargs):
        print ("=== Word Embedding ===")
        updateParams(Config, DefConfig, kwargs)
        self.Config = Config
        self.DefConfig = DefConfig
        test_path(Config, "model_path", "Wrong path to W2V model. Word Embedding can't be done.")
        if Config["need_create_model"] != "True":
            return
        test_path(Config, "data_corpus_path", "Wrong corpus path. W2V model can't be created.")
        try:
            self.epochs = int(self.Config["epochs_total"])
        except ValueError:
            raise ValueError("Wrong quantity of epochs for training. W2V model can't be created.")
        try:
            self.ndim = int(self.Config["vectors_dimension"])
        except ValueError:
            raise ValueError("Wrong size of resulting vectors. W2V model can't be created.")
        self.createW2VModel()

    def createW2VModel(self):
        sentences = []
        count = 0
        print("Start to create W2V model...")
        print("Get input data...")
        ds = datetime.datetime.now()
        with open(get_abs_path(self.Config, "data_corpus_path"), 'r', encoding='UTF-8') as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                count += 1
                words = [w for w in line.strip().split()]
                sentences.append(words)
        f.close()
        de = datetime.datetime.now()
        print("Got %d lines from file %s in %s"
              % (count, get_abs_path(self.Config, "data_corpus_path"), get_formatted_date(ds, de)))
        numpy.random.shuffle(sentences)

        logger = EpochLogger(self.epochs)
        w2v = Word2Vec(size=self.ndim, window=10, min_count=3, workers=10)
        ds = datetime.datetime.now()
        print("Build vocabulary...")
        w2v.build_vocab(sentences)
        de = datetime.datetime.now()
        print("Vocabulary is built in %s" % (get_formatted_date(ds, de)))
        print("Train model...")
        ds = datetime.datetime.now()
        w2v.train(sentences, epochs=int(self.Config["epochs_total"]), total_examples=len(sentences), callbacks=[logger])
        de = datetime.datetime.now()
        print("W2V model is completed in %s" % (get_formatted_date(ds, de)))

        created_model_path = get_abs_path(self.Config, "model_path")
        if self.Config["include_current_time_in_model_name"]:
            modelName = os.path.basename(created_model_path)
            dt = "-" + datetime.datetime.now().strftime("%Y-%b-%d-%H%M%S")
            pInd = modelName.rfind(".")
            if pInd > 0:
                modelName = modelName[:pInd] + dt + modelName[pInd:]
            else:
                modelName += dt
        finalPath = os.path.dirname(created_model_path) + "/" + modelName
        ds = datetime.datetime.now()
        w2v.wv.save_word2vec_format(finalPath, binary=False)
        de = datetime.datetime.now()
        print("W2V model %s is saved in the text format in %s\n" % (finalPath, get_formatted_date(ds, de)))

class EpochLogger(CallbackAny2Vec):
    def __init__(self, epochs):
        self.epoch = 1
        self.epochs = epochs

    def on_epoch_begin(self, model):
        print("Epoch %d from %d" % (self.epoch, self.epochs), end='\r')

    def on_epoch_end(self, model):
        self.epoch += 1
