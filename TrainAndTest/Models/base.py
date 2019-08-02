import gensim
import os
import shutil
import datetime
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from Models.metrics import ModelMetrics, metricsNames, print_metrics, print_averaged_metrics
from Models.dataPreparation import DataPreparation
from Utils.utils import align_to_left, get_formatted_date, get_abs_path, correct_path


class BaseModel:
    def __init__(self, Config):
        self.Config = Config
        self.trainArrays = []
        self.trainLabels = []
        self.testArrays = []
        self.testLabels = []
        self.valArrays = []
        self.valLabels = []
        self.cvDocs = []
        self.predictions = []
        self.metrics = {}
        self.resources = {}
        self.addValSet = False
        self.validation_data_size = 0
        self.isCV = False
        self.handleType = ""
        self.useProbabilities = False

        self.epochs = int(Config["epochs"])
        self.verbose = int(Config["verbose"])
        self.cross_validations_total = int(Config["cross_validations_total"])
        if self.verbose != 0:
            self.verbose = 1
        if Config["customrank"] == "True":
            self.rank_threshold = float(Config["rank_threshold"])
        else:
            self.rank_threshold = 0.5
        if self.rank_threshold == 0:
            self.rank_threshold = 0.5
        self.train_batch = int(Config["train_batch"])

    def is_correct_path(self, Config):
        if not correct_path(Config, "binarizer_path"):
            if Config["type_of_execution"] == "test" or Config["binarizer_path"]:
                print("Wrong path to binarizer. Stop.")
                return False
        if not correct_path(Config, "vectorizer_path"):
            if Config["type_of_execution"] == "test" or Config["vectorizer_path"]:
                print("Wrong path to vectorizer. Stop.")
                return False
        return True

    def launch_process(self):
        if self.Config["type_of_execution"] == "crossvalidation":
            self.isCV = True
            self.launchCrossValidation()
        elif self.Config["type_of_execution"] != "test":
            self.model = self.create_model()
            self.train_model()
            if self.Config["type_of_execution"] != "train":
                self.test_model()
        else:
            self.load_model()
            self.test_model()

    def create_model(self):
        pass

    def load_model(self):
        pass

    def train_model(self):
        pass

    def test_model(self):
        pass

    def load_w2v_model(self):
        if self.Config["w2vmodel"] != None:
            print ("W2V model is already loaded...")
            self.w2vModel = self.Config["w2vmodel"]
            return
        print ("Load W2V model... ")
        ds = datetime.datetime.now()
        self.w2vModel = gensim.models.KeyedVectors.load_word2vec_format(get_abs_path(self.Config, "model_path"))
        de = datetime.datetime.now()
        print("Load W2V model (%s) in %s" % (get_abs_path(self.Config, "model_path"), get_formatted_date(ds, de)))
        self.Config["resources"]["w2v"]["created_model_path"] = get_abs_path(self.Config, "model_path")
        self.Config["resources"]["w2v"]["ndim"] = self.ndim

    def loadNNModel(self):
        return load_model(get_abs_path(self.Config, "created_model_path", opt="name"))

    def loadSKLModel(self):
        return joblib.load(get_abs_path(self.Config, "created_model_path", opt="name"))

    def trainNNModel(self):
        checkpoints = []
        if self.save_intermediate_results and not self.isCV:
            checkpoint = ModelCheckpoint(get_abs_path(self.Config, "intermediate_results_path") + "/tempModel.hdf5",
                                         monitor='val_acc', verbose=self.verbose, save_best_only=True, mode='auto')
            checkpoints.append(checkpoint)
        print("Start training...              ")
        ds = datetime.datetime.now()
        self.model.fit(self.trainArrays, self.trainLabels, epochs=self.epochs,
                validation_data=(self.valArrays, self.valLabels),
                batch_size=self.train_batch, verbose=self.verbose, callbacks=checkpoints, shuffle=False)
        de = datetime.datetime.now()
        print("Model is trained in %s" %  (get_formatted_date(ds, de)))
        if self.isCV:
            return
        self.model.save(get_abs_path(self.Config, "created_model_path", opt="name"))
        print ("Model evaluation...")
        scores1 = self.model.evaluate(self.testArrays, self.testLabels, verbose=self.verbose)
        print("Final model accuracy: %.2f%%" % (scores1[1] * 100))
        if self.save_intermediate_results:
            model1 = load_model(get_abs_path(self.Config, "intermediate_results_path") + "/tempModel.hdf5")
            scores2 = model1.evaluate(self.testArrays, self.testLabels, verbose=self.verbose)
            print("Last saved model accuracy: %.2f%%" % (scores2[1] * 100))
            if scores1[1] < scores2[1]:
                model = model1
            pref = "The best model "
        else:
            pref = "Model "
        self.model.save(get_abs_path(self.Config, "created_model_path", opt="name"))
        print (pref + "is saved in %s" % get_abs_path(self.Config, "created_model_path", opt="name"))

    def trainSKLModel(self):
        de = datetime.datetime.now()
        print("Start training...")
        self.model.fit(self.trainArrays, self.trainLabels)
        ds = datetime.datetime.now()
        print("Model is trained in %s" % (get_formatted_date(de, ds)))
        if self.isCV:
            return
        joblib.dump(self.model, get_abs_path(self.Config, "created_model_path", opt="name"))
        print ("Model is saved in %s" % get_abs_path(self.Config, "created_model_path", opt="name"))
        print("Model evaluation...")
        prediction = self.model.predict(self.testArrays)
        print('Final accuracy is %.2f' % accuracy_score(self.testLabels, prediction))
        de = datetime.datetime.now()
        print("Evaluated in %s" % get_formatted_date(ds, de))

    def testNNModel(self):
        print ("Start testing...")
        print("Rank threshold: %.2f" % self.rank_threshold)
        ds = datetime.datetime.now()
        self.predictions = self.model.predict(self.testArrays)
        de = datetime.datetime.now()
        print("Test dataset containing %d documents predicted in %s\n" % (len(self.testArrays), get_formatted_date(ds, de)))
        if self.isCV:
            return
        self.prepare_resources_for_runtime("keras")
        self.get_metrics()
        self.saveResults()

    def testSKLModel(self):
        print ("Start testing...")
        if self.useProbabilities:
            print ("Rank threshold: %.2f" % self.rank_threshold)
        else:
            print("Model doesn't calculate probabilities.")
        ds = datetime.datetime.now()
        if not self.useProbabilities:
            self.predictions = self.model.predict(self.testArrays)
        else:
            self.predictions = self.model.predict(self.testArrays)
        de = datetime.datetime.now()
        print("Test dataset containing %d documents predicted in %s"
              % (self.testArrays.shape[0], get_formatted_date(ds, de)))
        if self.isCV:
            return
        self.prepare_resources_for_runtime("skl")
        self.get_metrics()
        self.saveResults()

    def get_metrics(self):
        print ("Calculate metrics...")
        ModelMetrics(self)
        if self.Config["show_test_results"] == "True":
            print_metrics(self)

    def saveResults(self):
        self.Config["results"][self.Config["name"]] = self.predictions
        self.Config["metrics"][self.Config["name"]] = self.metrics
        if self.useProbabilities:
            self.Config["ranks"][self.Config["name"]] = self.rank_threshold
        else:
            self.Config["ranks"][self.Config["name"]] = 1.0

    def prepare_resources_for_runtime(self, type):
        self.resources["created_model_path"] = get_abs_path(self.Config, "created_model_path", opt="name")
        self.resources["modelType"] = type
        if self.useProbabilities:
            self.resources["rank_threshold"] = self.rank_threshold
        else:
            self.resources["rank_threshold"] = 1.0
        self.saveAdditions()
        if type == "skl":
            self.resources["handleType"] = "vectorize"
        self.Config["resources"]["models"]["Model" + str(self.Config["modelid"])] = self.resources

    def saveAdditions(self):
        pass

    def launchCrossValidation(self):
        print ("Start cross-validation...")
        ds = datetime.datetime.now()
        dp = DataPreparation(self, self.addValSet)
        pSize = len(self.cvDocs) // self.cross_validations_total
        ind = 0
        f1 = 0
        attr_metrics =[]
        for i in range(self.cross_validations_total):
            print ("Cross-validation, cycle %d from %d..." % ((i+1), self.cross_validations_total))
            if i == 0:
                self.Config["cross_validations_train_docs"] = self.cvDocs[pSize:]
                self.Config["cross_validations_test_docs"] = self.cvDocs[:pSize]
            elif i == self.cross_validations_total - 1:
                self.Config["cross_validations_train_docs"] = self.cvDocs[:ind]
                self.Config["cross_validations_test_docs"] = self.cvDocs[ind:]
            else:
                self.Config["cross_validations_train_docs"] = self.cvDocs[:ind] + self.cvDocs[ind+pSize:]
                self.Config["cross_validations_test_docs"] = self.cvDocs[ind:ind+pSize]
            ind += pSize
            dp.getVectors(self.handleType)
            self.model = self.create_model()
            self.train_model()
            self.test_model()
            ModelMetrics(self)
            attr_metrics.append(self.metrics)
            cycleF1 = self.metrics["all"]["f1"]
            print ("Resulting F1-Measure: %f\n" % cycleF1)
            if cycleF1 > f1:
                if self.Config["save_cross_validations_datasets"]:
                    self.saveDataSets()
                f1 = cycleF1
        de = datetime.datetime.now()
        print ("Cross-validation is done in %s" % get_formatted_date(ds, de))
        print_averaged_metrics(attr_metrics, self.Config)
        print ("The best result is %f"%(f1))
        print ("Corresponding data sets are saved in the folder %s"
               % get_abs_path(self.Config, "cross_validations_datasets_path"))


    def saveDataSets(self):
        root = get_abs_path(self.Config, "cross_validations_datasets_path")
        shutil.rmtree(root)
        os.mkdir(root)
        train_data_path = root + "/train"
        test_data_path = root + "/test"
        folds = {}
        os.mkdir(train_data_path)
        for doc in self.Config["cross_validations_train_docs"]:
            for nlab in doc.nlabs:
                foldPath = train_data_path + "/" + nlab
                if nlab not in folds:
                    os.mkdir(foldPath)
                    folds[nlab] = True
                with open(foldPath + '/' + doc.name, 'w', encoding="utf-8") as file:
                    file.write(doc.lines)
                file.close()
        folds = {}
        os.mkdir(test_data_path)
        for doc in self.Config["cross_validations_test_docs"]:
            for nlab in doc.nlabs:
                foldPath = test_data_path + "/" + nlab
                if nlab not in folds:
                    os.mkdir(foldPath)
                    folds[nlab] = True
                with open(foldPath + '/' + doc.name, 'w', encoding="utf-8") as file:
                    file.write(doc.lines)
                file.close()
