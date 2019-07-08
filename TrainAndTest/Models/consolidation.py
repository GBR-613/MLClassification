import os
import numpy
import shutil
import pickle
import json
import datetime
from Models.metrics import ModelMetrics, printMetrics
from Utils.utils import get_absolute_path, show_time
from Models.reports import Report

class Collector:
    def __init__(self, Config):
        self.Config = Config
        if "test_docs" not in Config or len(Config["results"]) == 0:
            print ("Documents have not been classified in this process chain.")
            print ("Consolidation can't be performed.")
            self.rank_threshold = 0.5
            if Config['consolidatedrank'] == "yes":
                try:
                    self.rank_threshold = float(Config["consolidated_rank_threshold"])
                except ValueError:
                    self.rank_threshold = 0.5
            return
        self.testLabels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1,
                                        len(self.Config["predefined_categories"])) for x in self.Config["test_docs"]])
        self.qLabs = len(self.Config["predefined_categories"])
        self.predictions = numpy.zeros([len(self.testLabels), self.qLabs])
        self.metrics = {}
        self.useProbabilities = False
        self.save_reports = False
        self.runtime = False
        print ("\nCalculate consolidated metrics...")
        if len(self.Config["results"]) == 0:
            print("No results to consolidate them.")
            print("Consolidation can't be performed.")
            return
        if Config["save_reports"] == "True":
            if len(Config["reports_path"]) == 0 or not os.path.isdir(get_absolute_path(Config, "reports_path")):
                print("Wrong path to the folder, containing reports.")
                print("Reports can't be created.")
            else:
                self.save_reports = True
        if Config["prepare_resources_for_runtime"] == "True":
            if (len(Config["saved_resources_path"]) == 0 or
                    not os.path.isdir(get_absolute_path(Config, "saved_resources_path"))):
                print("Wrong path to the folder, containing resources for runtime.")
                print("Resources can't be saved.")
            else:
                self.runtime = True
        print("Rank threshold for consolidated results: %.2f" % (self.rank_threshold))
        if self.save_reports or self.Config["show_consolidated_results"] == "True":
            self.getConsolidatedResults()
            self.getMetrics()
            if self.save_reports:
                self.saveReports()
        if self.runtime:
            if len(os.listdir(get_absolute_path(self.Config, "saved_resources_path"))) > 0:
                print("Warning: folder %s is not empty. All its content will be deleted."%(
                                get_absolute_path(self.Config, "saved_resources_path")))
                shutil.rmtree(get_absolute_path(self.Config, "saved_resources_path"))
                os.makedirs(get_absolute_path(self.Config, "saved_resources_path"), exist_ok=True)
            print("\nCollect arfifacts for runtime...")
            self.prepare_resources_for_runtime()


    def getConsolidatedResults(self):
        for key, res in self.Config["results"].items():
            for i in range(len(res)):
                for j in range(self.qLabs):
                    if res[i][j] == 1:
                        self.predictions[i][j] += 1
                    #elif res[i][j] >= self.rank_threshold:
                    elif res[i][j] >= self.Config["ranks"][key]:
                        self.predictions[i][j] += 1
        qModels = len(self.Config["results"])
        for i in range(len(self.predictions)):
            for j in range(len(self.predictions[i])):
                if self.predictions[i][j] >= qModels * self.rank_threshold:
                    self.predictions[i][j] = 1
                else:
                    self.predictions[i][j] = 0

    def getMetrics(self):
        ModelMetrics(self)
        if self.Config["show_consolidated_results"] == "True":
            printMetrics(self)

    def saveReports(self):
        print ("Save report...")
        report = Report()
        report.requestId = self.Config["reqid"]
        report.sourcesPath = self.Config["actualpath"]
        report.datasetPath = self.Config["test_data_path"]

        tokOpts = ["language_tokenization", "normalization", "stopwords",
                   "exclude_positions", "extrawords", "exclude_categories"]
        for i in range(len(tokOpts)):
            report.preprocess[tokOpts[i]] = self.Config[tokOpts[i]]
        for i in range(len(self.Config["test_docs"])):
            report.docs[self.Config["test_docs"][i].name] = {}
            report.docs[self.Config["test_docs"][i].name]["actual"] = ",".join(self.Config["test_docs"][i].nlabs)
        if len(self.Config["exclude_categories"]) == 0:
            exclude_categories = []
        else:
            exclude_categories = self.Config["exclude_categories"].split(",")
        cNames = [''] * (len(self.Config["predefined_categories"]) - len(exclude_categories))
        for k, v in self.Config["predefined_categories"].items():
            if k not in exclude_categories:
                cNames[v] = k
        report.categories = cNames
        for key, val in self.Config["results"].items():
            for i in range(len(val)):
                labs = []
                for j in range(self.qLabs):
                    #if val[i][j] >= self.rank_threshold:
                    if val[i][j] >= self.Config["ranks"][key]:
                        labs.append("%s[%.2f]" % (cNames[j], val[i][j]))
                report.docs[self.Config["test_docs"][i].name][key] = ",".join(labs)
        for key, val in self.Config["metrics"].items():
            report.models[key] = val
        for key, val in self.Config["ranks"].items():
            report.ranks[key] = val
        if len(self.Config["results"]) > 1:
            for i in range(len(self.predictions)):
                labs = []
                for j in range(self.qLabs):
                    if self.predictions[i][j] == 1:
                        labs.append(cNames[j])
                report.docs[self.Config["test_docs"][i].name]["consolidated"] = ",".join(labs)
            report.models["consolidated"] = self.rank_threshold
        rPath = get_absolute_path(self.Config, "reports_path") + "/" + self.Config["reqid"] + ".json"
        with open(rPath, 'w', encoding="utf-8") as file:
            json.dump(report.toJSON(), file, indent=4)
        file.close()

    def prepare_resources_for_runtime(self):
        tokOpts = ["language_tokenization", "normalization", "stopwords", "exclude_positions", "extrawords",
                   "maxseqlen", "maxcharsseqlen", "single_doc_lang_tokenization_lib_path"]
        self.Config["resources"]["tokenization"] = {}
        ds = datetime.datetime.now()
        self.outDir = get_absolute_path(self.Config, "saved_resources_path") + "/"
        for i in range(len(tokOpts)):
            if tokOpts[i] != "single_doc_lang_tokenization_lib_path":
                self.Config["resources"]["tokenization"][tokOpts[i]] = self.Config[tokOpts[i]]
            elif self.Config["language_tokenization"] == "True":
                self.Config["resources"]["tokenization"]["single_doc_lang_tokenization_lib_path"] = \
                    self.copyFile(get_absolute_path(self.Config, "single_doc_lang_tokenization_lib_path"))
        isW2VNeeded = False
        for key, val in self.Config["resources"]["models"].items():
            val["created_model_path"] = self.copyFile(val["created_model_path"])
            if "w2v" in val and val["w2v"] == "True":
                isW2VNeeded = True
        if not isW2VNeeded and "w2v" in self.Config["resources"]:
            self.Config["resources"].pop("w2v", None)
        if "w2v" in self.Config["resources"]:
            w2vDict = {}
            isFirstLine = True
            fEmbeddings = open(self.Config["resources"]["w2v"]["created_model_path"], encoding="utf-8")
            for line in fEmbeddings:
                if isFirstLine == True:
                    isFirstLine = False
                    continue
                split = line.strip().split(" ")
                word = split[0]
                vector = numpy.array([float(num) for num in split[1:]])
                w2vDict[word] = vector
            fEmbeddings.close()
            with open(self.Config["resources"]["w2v"]["created_model_path"] + '.pkl', 'wb') as file:
                pickle.dump(w2vDict, file, pickle.HIGHEST_PROTOCOL)
            file.close()
            self.Config["resources"]["w2v"]["created_model_path"] = \
                self.copyFile(self.Config["resources"]["w2v"]["created_model_path"] + '.pkl')
        if "indexer" in self.Config["resources"]:
            self.Config["resources"]["indexer"] = self.copyFile(self.Config["resources"]["indexer"])
        if "vectorizer" in self.Config["resources"]:
            self.Config["resources"]["vectorizer"] = self.copyFile(self.Config["resources"]["vectorizer"])
        if "ptBertModel" in self.Config["resources"]:
            self.Config["resources"]["ptBertModel"] = self.copyFile(self.Config["resources"]["ptBertModel"])
            self.Config["resources"]["vocabPath"] = self.copyFile(self.Config["resources"]["vocabPath"])
        cNames = [''] * len(self.Config["predefined_categories"])
        for k, v in self.Config["predefined_categories"].items():
            cNames[v] = k
        with open(self.outDir + 'labels.txt', 'w', encoding="utf-8") as file:
            file.write(",".join(cNames))
        file.close()
        self.Config["resources"]["labels"] = "labels.txt"
        self.Config["resources"]["consolidatedRank"] = self.rank_threshold
        with open(self.outDir + 'config.json', 'w', encoding="utf-8") as file:
            json.dump(self.Config["resources"], file, indent=4)
        file.close()
        de = datetime.datetime.now()
        print("\nArtifacts are copied into the folder %s in %s"%(
            get_absolute_path(self.Config, "saved_resources_path"), show_time(ds, de)))

    def copyFile(self, inPath):
        dir, name = os.path.split(inPath)
        outPath = self.outDir + name
        shutil.copy(inPath, outPath)
        return name

