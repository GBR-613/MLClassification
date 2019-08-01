import os
from Utils.utils import get_absolute_path, updateParams, correct_path, test_path
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


class ModelController:
    def __init__(self, Config, DefConfig, kwargs):
        Config["modelid"] += 1
        print ("=== Model " + str(Config["modelid"]) + " ===")
        updateParams(Config, DefConfig, kwargs)
        Config["type"] = Config["type"].lower()
        Config["type_of_execution"] = Config["type_of_execution"].lower()
        if Config["type_of_execution"] != "none" and Config["type"] not in modelTypes:
            raise ValueError("Request contains definition of model with wrong type. Stop.")
        if Config["type_of_execution"] not in modelGoals:
            raise ValueError("Request doesn't define the goal of the model process. "
                             "It should be one of 'trainAndTest', 'train', 'test', 'crossValidation' or 'none'. Stop.")
        if Config["type_of_execution"] != "none":
            print ("Model type: " + Config["type"].upper() + ", " + userInfo[Config["type_of_execution"]])
        else:
            print("Model : " +  userInfo[Config["type_of_execution"]])
        if Config["type_of_execution"] == "none":
            return
        self.Config = Config
        self.DefConfig = DefConfig
        if "predefined_categories" not in Config or "train_docs" not in Config or "test_docs" not in Config:
            raise ValueError("Input data isn't loaded. Stop.")

        try:
            self.test_data_size = float(Config["test_data_size"])
        except ValueError:
            self.test_data_size = -1
        if not correct_path(Config, "train_data_path"):
            if Config["type_of_execution"] != "test" or not Config["test_data_path"]:
                raise ValueError("Wrong path to the training set: folder %s doesn't exist."%(get_absolute_path(Config, "train_data_path")))
        if not correct_path(Config, "test_data_path"):
            if not (len(Config["test_data_path"]) == 0 and self.test_data_size > 0 and self.test_data_size < 1):
                raise ValueError("Wrong path to the testing set: folder %d doesn't exist."%(get_absolute_path(Config, "test_data_path")))
        test_path(Config, "created_model_path", "Wrong path to the models' folder.")
        if not Config["name"]:
            Config["name"] = Config["type"] + str(Config["modelid"])
        mPath = get_absolute_path(Config, "created_model_path", opt="name")
        if Config["type_of_execution"] == "test" and not os.path.isfile(mPath):
            raise ValueError("Wrong path to the tested model.")
        if Config["type_of_execution"] != "test":
            try:
                self.epochs = int(Config["epochs"])
            except ValueError:
                raise ValueError("Wrong quantity of epochs for training.")
            try:
                self.train_batch = int(Config["train_batch"])
            except ValueError:
                raise ValueError("Wrong batch size for training.")
            try:
                self.verbose = int(Config["verbose"])
            except ValueError:
                raise ValueError("Wrong value of 'verbose' flag for training.")
            if Config["save_intermediate_results"] == "True":
                if not Config["intermediate_results_path"] or \
                        not os.path.isdir(get_absolute_path(Config, "intermediate_results_path")):
                    raise ValueError("Wrong path to folder with intermediate results.")
        """
        if Config["type_of_execution"].lower() != "train":
            if Config["modelinfo"] == "True":
                if not Config["infopath"] or not os.path.isdir(get_absolute_path(Config, "infopath")):
                    raise ValueError("Wrong path to folder containing model info.")
        """
        if Config["type_of_execution"] != "train" and Config["customrank"] == "True":
            try:
                self.rank_threshold = float(Config["rank_threshold"])
            except ValueError:
                raise ValueError("Wrong custom rank threshold.")
        if Config["type_of_execution"] == "crossvalidation":
            if Config["save_cross_validations_datasets"] == "True":
                test_path(Config, "cross_validations_datasets_path",
                          "Wrong path to the cross-validation's resulting folder.")
            try:
                cross_validations_total = int(Config["cross_validations_total"])
            except ValueError:
                raise ValueError("Wrong k-fold value.")
        #if stop:
        #    print ("Stop.")
        #    Config["error"] = True
        #    return
        if Config["type"].lower() == "snn":
            SnnModel(Config)
        elif Config["type"].lower() == "ltsm":
            LTSMModel(Config)
        elif Config["type"].lower() == "cnn":
            CNNModel(Config)
        elif Config["type"].lower() == "pac":
            PacModel(Config)
        elif Config["type"].lower() == "ridge":
            RidgeModel(Config)
        elif Config["type"].lower() == "svc":
            SVCModel(Config)
        elif Config["type"] == "perceptron":
            PerceptronModel(Config)
        elif Config["type"] == "sgd":
            SGDModel(Config)
        elif Config["type"] == "bert":
            BertModel(Config)
