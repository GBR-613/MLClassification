import os
from Utils.utils import get_abs_path, updateParams, correct_path, test_path
from Models.snn import SnnModel
from Models.ltsm import LTSMModel
from Models.cnn import CNNModel
from Models.pac import PacModel
from Models.ridge import RidgeModel
from Models.svc import SVCModel
from Models.perceptron import PerceptronModel
from Models.sgd import SGDModel
from Models.bert import BertModel

modelTypes = ["snn", "ltsm", "cnn", "pac", "perceptron", "ridge", "sgd", "svc", "bert"]
modelGoals = ["trainandtest", "train", "test", "crossvalidation", "none"]
userInfo = {
    "trainandtest": "training and testing",
    "train": "training only",
    "test": "testing only",
    "crossvalidation": "cross-validation",
    "none": "set parameters only"}


def ModelController(Config, DefConfig, kwargs):
    worker = _ModelController(Config, DefConfig, kwargs)
    worker.run()


class _ModelController:
    def __init__(self, Config, DefConfig, kwargs):
        self.Config = Config
        self.DefConfig = DefConfig
        self.Config["modelid"] += 1
        print ("=== Model " + str(self.Config["modelid"]) + " ===")
        updateParams(self.Config, DefConfig, kwargs)
        self.Config["type"] = self.Config["type"].lower()
        self.Config["type_of_execution"] = self.Config["type_of_execution"].lower()
        if self.Config["type_of_execution"] != "none" and self.Config["type"] not in modelTypes:
            raise ValueError("Request contains definition of model with wrong type. Stop.")
        if self.Config["type_of_execution"] not in modelGoals:
            raise ValueError("Request doesn't define the goal of the model process. "
                             "It should be one of 'trainAndTest', 'train', 'test', 'crossValidation' or 'none'. Stop.")
        if self.Config["type_of_execution"] != "none":
            print ("Model type: " + self.Config["type"].upper() + ", " + userInfo[self.Config["type_of_execution"]])
        else:
            print("Model : " +  userInfo[self.Config["type_of_execution"]])
        if self.Config["type_of_execution"] == "none":
            return
        if "predefined_categories" not in self.Config or "train_docs" not in self.Config or "test_docs" not in self.Config:
            raise ValueError("Input data isn't loaded. Stop.")
        
    def run(self):
        try:
            self.test_data_size = float(self.Config["test_data_size"])
        except ValueError:
            self.test_data_size = -1
        if not correct_path(self.Config, "train_data_path"):
            if self.Config["type_of_execution"] != "test" or not self.Config["test_data_path"]:
                raise ValueError("Wrong path to the training set: folder %s doesn't exist."
                                 % get_abs_path(self.Config, "train_data_path"))
        if not correct_path(self.Config, "test_data_path"):
            if not (len(self.Config["test_data_path"]) == 0 and self.test_data_size > 0 and self.test_data_size < 1):
                raise ValueError("Wrong path to the testing set: folder %d doesn't exist."
                                 % get_abs_path(self.Config, "test_data_path"))
        test_path(self.Config, "created_model_path", "Wrong path to the models' folder.")
        if not self.Config["name"]:
            self.Config["name"] = self.Config["type"] + str(self.Config["modelid"])
        mPath = get_abs_path(self.Config, "created_model_path", opt="name")
        if self.Config["type_of_execution"] == "test" and not os.path.isfile(mPath):
            raise ValueError("Wrong path to the tested model.")
        if self.Config["type_of_execution"] != "test":
            try:
                self.epochs = int(self.Config["epochs"])
            except ValueError:
                raise ValueError("Wrong quantity of epochs for training.")
            try:
                self.train_batch = int(self.Config["train_batch"])
            except ValueError:
                raise ValueError("Wrong batch size for training.")
            try:
                self.verbose = int(self.Config["verbose"])
            except ValueError:
                raise ValueError("Wrong value of 'verbose' flag for training.")
            if self.Config["save_intermediate_results"] == "True":
                if not self.Config["intermediate_results_path"] or \
                        not os.path.isdir(get_abs_path(self.Config, "intermediate_results_path")):
                    raise ValueError("Wrong path to folder with intermediate results.")
        """
        if self.Config["type_of_execution"].lower() != "train":
            if self.Config["modelinfo"] == "True":
                if not self.Config["infopath"] or not os.path.isdir(get_abs_path(self.Config, "infopath")):
                    raise ValueError("Wrong path to folder containing model info.")
        """
        if self.Config["type_of_execution"] != "train" and self.Config["customrank"] == "True":
            try:
                self.rank_threshold = float(self.Config["rank_threshold"])
            except ValueError:
                raise ValueError("Wrong custom rank threshold.")
        if self.Config["type_of_execution"] == "crossvalidation":
            if self.Config["save_cross_validations_datasets"] == "True":
                test_path(self.Config, "cross_validations_datasets_path",
                          "Wrong path to the cross-validation's resulting folder.")
            try:
                cross_validations_total = int(self.Config["cross_validations_total"])
            except ValueError:
                raise ValueError("Wrong k-fold value.")
        #if stop:
        #    print ("Stop.")
        #    self.Config["error"] = True
        #    return
        if self.Config["type"].lower() == "snn":
            SnnModel(self.Config)
        elif self.Config["type"].lower() == "ltsm":
            LTSMModel(self.Config)
        elif self.Config["type"].lower() == "cnn":
            CNNModel(self.Config)
        elif self.Config["type"].lower() == "pac":
            PacModel(self.Config)
        elif self.Config["type"].lower() == "ridge":
            RidgeModel(self.Config)
        elif self.Config["type"].lower() == "svc":
            SVCModel(self.Config)
        elif self.Config["type"] == "perceptron":
            PerceptronModel(self.Config)
        elif self.Config["type"] == "sgd":
            SGDModel(self.Config)
        elif self.Config["type"] == "bert":
            BertModel(self.Config)
