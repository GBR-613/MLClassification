import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Utils.utils import get_abs_path


class SGDModel(BaseModel):
    def __init__(self, Config):
        super().__init__(Config)
        if not self.is_correct_path(Config):
            raise Exception
        self.useProbabilities = True
        self.handleType = "vectorize"
        if Config["type_of_execution"] != "crossvalidation":
            self.prepareData()
        self.launch_process()

    def prepareData(self):
        print("Start data preparation...")
        dp = DataPreparation(self, False)
        dp.getDataForSklearnClassifiers()

    def create_model(self):
        return OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='elasticnet',
                                                 alpha=1e-4, max_iter=10, tol=1e-3, n_jobs=-1))

    def load_model(self):
        self.model = self.loadSKLModel()

    def train_model(self):
        self.trainSKLModel()

    def test_model(self):
        self.testSKLModel()

    def save_additions(self):
        if not "vectorizer" in self.Config["resources"]:
            self.Config["resources"]["vectorizer"] = get_abs_path(self.Config, "vectorizer_path")
        self.resources["vectorizer"] = "True"

