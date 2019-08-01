import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import Perceptron
from sklearn.externals import joblib
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Utils.utils import get_absolute_path


class PerceptronModel(BaseModel):
    def __init__(self, Config):
        super().__init__(Config)
        if not self.isCorrectPath(Config):
            raise Exception
        self.useProbabilities = False
        self.handleType = "vectorize"
        if Config["type_of_execution"] != "crossvalidation":
            self.prepareData()
        self.launchProcess()

    def prepareData(self):
        print("Start data preparation...")
        dp = DataPreparation(self, False)
        dp.getDataForSklearnClassifiers()

    def createModel(self):
        return OneVsRestClassifier(Perceptron(n_jobs=-1, max_iter=20, tol=1e-3))

    def loadModel(self):
        self.model = self.loadSKLModel()

    def trainModel(self):
        self.trainSKLModel()

    def testModel(self):
        #self.model = joblib.load(get_absolute_path(self.Config, "created_model_path", opt="name"))
        self.testSKLModel()

    def saveAdditions(self):
        if not "vectorizer" in self.Config["resources"]:
            self.Config["resources"]["vectorizer"] = get_absolute_path(self.Config, "vectorizer_path")
        self.resources["vectorizer"] = "True"
