import os
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Models.metrics import ModelMetrics
from Utils.utils import get_abs_path, get_formatted_date, test_path, correct_path


class LTSMModel(BaseModel):
    def __init__(self, Config):
        super().__init__(Config)
        if not self.is_correct_path(Config):
            raise Exception
        try:
            self.validation_data_size = float(Config["validation_data_size"])
        except ValueError:
            self.validation_data_size = 0
        if self.validation_data_size <= 0 or self.validation_data_size >= 1:
            raise ValueError("Wrong size of validation data set. Stop.")
        try:
            self.ndim = int(self.Config["vectors_dimension"])
        except ValueError:
            raise ValueError("Wrong size of vectors' dimentions. Stop.")
        self.addValSet = True
        self.handleType = "wordVectorsMatrix"
        self.save_intermediate_results = Config["save_intermediate_results"] == "True"
        self.useProbabilities = True
        self.w2vModel = None
        self.load_w2v_model()
        if Config["type_of_execution"] != "crossvalidation":
            self.prepareData()
        self.launch_process()

    def is_correct_path(self, Config):
        if self.Config["w2vmodel"] == None:
            test_path(Config, "model_path", "Wrong path to W2V model. Stop.")
        if not correct_path(Config, "indexer_path"):
            if Config["type_of_execution"] == "test":
                print("Wrong path to indexer. Stop.")
                return False
        return True

    def prepareData(self):
        print ("Start data preparation...")
        dp = DataPreparation(self, self.addValSet)
        self.embMatrix, self.maxWords = dp.getWordVectorsMatrix()

    def create_model(self):
        model = Sequential()
        model.add(Embedding(self.maxWords, self.ndim, input_length=self.Config["max_seq_len"]))
        model.layers[0].set_weights([self.embMatrix])
        model.layers[0].trainable = False
        model.add(LSTM(self.Config["max_seq_len"]))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(len(self.Config["predefined_categories"]), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def load_model(self):
        self.model = self.loadNNModel()

    def train_model(self):
        self.trainNNModel()

    def test_model(self):
        self.testNNModel()

    def saveAdditions(self):
        self.resources["w2v"] = "True"
        if not "indexer" in self.Config["resources"]:
            self.Config["resources"]["indexer"] = get_abs_path(self.Config, "indexer_path")
        self.resources["indexer"] = "True"
        self.resources["handleType"] = "wordVectorsMatrix"
