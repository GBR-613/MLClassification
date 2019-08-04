import os
import datetime
from pathlib import Path
from configparser import ConfigParser
from Preprocess.preprocess import job_preprocessor
from WordEmbedding.vectors import job_word_embedding
from Data.data import job_data_loader
from Models.controller import job_model_controller
from Models.consolidation import job_collector
from Utils.utils import get_configuration, test_path
from Info.creator import InfoCreator

Config = {}
parser = ConfigParser()
jobs_list = []

jobs_def = {
    "P": (job_preprocessor,"preprocess"),
    "W": (job_word_embedding,"word_embedding"),
    "D": (job_data_loader,"data"),
    "M": (job_model_controller,"model"),
    "C": (job_collector,"")
}


def parse_config(path):
    parser.read_file(open(path))
    for s in parser.sections():
        for opt in parser.items(s):
            Config[opt[0]] = opt[1]
    if not Config["home"]:
        Config["home"] = str(Path.home())
    Config["reqid"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    Config["modelid"] = 0
    Config["results"] = {}
    Config["metrics"] = {}
    Config["ranks"] = {}
    Config["resources"] = {}
    Config["resources"]["reqid"] = Config["reqid"]
    Config["resources"]["models"] = {}
    Config["resources"]["w2v"] = {}


def parse_request(req):
    print("=== Request " + Config["reqid"] + " ===")
    print(req)
    if not req:
        req = (Config["request"]).strip().replace(" ", "")
    if not req:
        raise ValueError("Request is not defined, nothing to do")
    tasks = req.split("|")
    for task in tasks:
        task_name = task[0]
        if task_name not in jobs_def.keys():
            raise ValueError("Wrong task name, should be one of P,W,M,D,C: " + task_name)
        if not (task[1] == "(" and task[-1] == ")"):
            raise ValueError("Wrong definition of task name ('%s'). Exit." % task)
        definition = task[2:-1]
        kwargs = {}
        if definition != "":
            for option in definition.split(";"):
                kvs = option.split("=")
                if kvs[0].lower() not in Config:
                    raise ValueError("Wrong parameter ('%s') of task name '%s'. Stop." % (kvs[0], task_name))
                for k in range(len(kvs)):
                    kwargs[kvs[0].lower()] = kvs[1]
        jobs_list.append((task_name, kwargs))


def work():
    for job in jobs_list:
        print(datetime.datetime.now())
        print(" Start task " + job[0])
        func = jobs_def[job[0]][0]
        kwargs = job[1]
        job_config_name = jobs_def[job[0]][1]
        job_config = get_configuration(parser, job_config_name)
        func(Config, job_config, kwargs)


def parse_config_info(path):
    parser.read_file(open(path))
    for s in parser.sections():
        for opt in parser.items(s):
            Config[opt[0]] = opt[1]
    if not Config["home"]:
        Config["home"] = str(Path.home())
    if not Config["info_from"]:
        Config["info_from"] = "today"
    if Config["info_from"] != "today":
        chk = Config["info_from"].split()
        if len(chk) != 2 and not chk[1].startswith("day"):
            print ("Wrong value of 'info_from' option. Exit.")
            return
        try:
            days = int(chk[0])
        except ValueError:
            print ("Wrong value of 'info_from' option. Exit.")
            return
    test_path(Config, "reports_path", "Wrong path to the folder, containing reports. Exit.")
    test_path(Config, "actual_path",
              "Warning: wrong path to the folder containing original documents. It will not be possible to view them.")
    InfoCreator(Config)


