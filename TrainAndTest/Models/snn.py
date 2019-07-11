import os
import gensim
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Models.metrics import ModelMetrics
from Utils.utils import get_absolute_path, show_time, test_path

class SnnModel(BaseModel):
    def __init__(self, Config):
        super().__init__(Config)
        if self.Config["w2vmodel"] == None:
            test_path(Config, "model_path", "Wrong path to W2V model. Stop.")
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
        self.handleType = "wordVectorsSum"
        self.save_intermediate_results = Config["save_intermediate_results"] == "True"
        self.useProbabilities = True
        self.w2vModel = None
        self.load_w2v_model()
        if Config["type_of_execution"] != "crossvalidation":
            self.prepareData()
        self.launchProcess()

    def prepareData(self):
        print ("Start data preparation...")
        dp = DataPreparation(self, self.addValSet)
        dp.getWordVectorsSum()

    def createModel(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=self.ndim))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.Config["predefined_categories"]), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def loadModel(self):
        self.model = self.loadNNModel()

    def trainModel(self):
        self.trainNNModel()

    def testModel(self):
        self.testNNModel()

    def saveAdditions(self):
        self.resources["w2v"] = "True"
        self.resources["handleType"] = "wordVectorsSum"