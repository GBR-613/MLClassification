import os
import numpy
import shutil
import pickle
import json
import datetime
from Models.metrics import ModelMetrics, print_metrics
from Utils.utils import get_abs_path, get_formatted_date
from Models.reports import Report


def job_collector(Config, DefConfig, kwargs):
    worker = Collector(Config, DefConfig, kwargs)
    worker.run()


class Collector:
    def __init__(self, Config, DefConfig, kwargs):
        self.Config = Config
        if "test_docs" not in Config or not Config["results"]:
            print ("Documents have not been classified in this process chain.")
            print ("Consolidation can't be performed.")
            self.rank_threshold = 0.5
            if Config['consolidatedrank'] == "True":
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

    def run(self):
        print ("\nCalculate consolidated metrics...")
        if not self.Config["results"]:
            print("No results to consolidate them. Consolidation can not be performed.")
            return
        if self.Config["save_reports"] == "True":
            if not self.Config["reports_path"] or not os.path.isdir(get_abs_path(self.Config, "reports_path")):
                print("Wrong path to the folder, containing reports. Reports can not be created.")
            else:
                self.save_reports = True
        if self.Config["prepare_resources_for_runtime"] == "True":
            if (not self.Config["saved_resources_path"] or
                    not os.path.isdir(get_abs_path(self.Config, "saved_resources_path"))):
                print("Wrong path to the folder, containing resources for runtime. Resources can not be saved.")
            else:
                self.runtime = True
        print("Rank threshold for consolidated results: %.2f" % (self.rank_threshold))
        if self.save_reports or self.Config["show_consolidated_results"] == "True":
            self.getConsolidatedResults()
            self.get_metrics()
            if self.save_reports:
                self.saveReports()
        if self.runtime:
            saved_rc_path = get_abs_path(self.Config, "saved_resources_path")
            if len(os.listdir(saved_rc_path)) > 0:
                print("Warning: folder %s is not empty. All its content will be deleted." % saved_rc_path)
                shutil.rmtree(saved_rc_path)
                os.makedirs(saved_rc_path, exist_ok=True)
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
        q_models = len(self.Config["results"])
        for p1 in self.predictions:
            for p in p1:
                if p >= q_models * self.rank_threshold:
                    p = 1
                else:
                    p = 0

    def get_metrics(self):
        ModelMetrics(self)
        if self.Config["show_consolidated_results"] == "True":
            print_metrics(self)

    def saveReports(self):
        print ("Save report...")
        report = Report()
        report.requestId = self.Config["reqid"]
        report.sourcesPath = self.Config["actual_path"]
        report.datasetPath = self.Config["test_data_path"]

        tokenization_options = ["language_tokenization", "normalization", "stop_words", "exclude_positions",
                                "extra_words", "exclude_categories"]
        for t in tokenization_options:
            report.preprocess[t] = self.Config[t]
        for t in self.Config["test_docs"]:
            report.docs[t.name] = {}
            report.docs[t.name]["actual"] = ",".join(t.nlabs)
        if not self.Config["exclude_categories"]:
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
        rPath = get_abs_path(self.Config, "reports_path") + "/" + self.Config["reqid"] + ".json"
        with open(rPath, 'w', encoding="utf-8") as file:
            json.dump(report.toJSON(), file, indent=4)
        file.close()

    def prepare_resources_for_runtime(self):
        tokenization_options = ["language_tokenization", "normalization", "stop_words", "exclude_positions",
                        "extra_words", "max_seq_len", "max_chars_seq_len", "single_doc_lang_tokenization_lib_path"]
        self.Config["resources"]["tokenization"] = {}
        ds = datetime.datetime.now()
        self.outDir = get_abs_path(self.Config, "saved_resources_path") + "/"
        for t in tokenization_options:
            if t != "single_doc_lang_tokenization_lib_path":
                self.Config["resources"]["tokenization"][t] = self.Config[t]
            elif self.Config["language_tokenization"] == "True":
                self.Config["resources"]["tokenization"]["single_doc_lang_tokenization_lib_path"] = \
                    self.copyFile(get_abs_path(self.Config, "single_doc_lang_tokenization_lib_path"))
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
            get_abs_path(self.Config, "saved_resources_path"), get_formatted_date(ds, de)))

    def copyFile(self, inPath):
        dir, name = os.path.split(inPath)
        outPath = self.outDir + name
        shutil.copy(inPath, outPath)
        return name

