import os
import datetime
from pathlib import Path
from configparser import ConfigParser
from Preprocess.preprocess import Preprocessor
from WordEmbedding.vectors import Embedding
from Data.data import DataLoader
from Models.controller import ModelController
from Models.consolidation import Collector
from Utils.utils import get_configuration, get_abs_path, test_path
from Info.creator import InfoCreator

Config = {}
parser = ConfigParser()
actions_list = []

actions_def = {
    "P": (Preprocessor,"preprocess"),
    "W": (Embedding,"word_embedding"),
    "D": (DataLoader,"data"),
    "M": (ModelController,"model"),
    "C": (Collector,"")
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
        if task_name not in actions_def.keys():
            raise ValueError("Wrong task name, should be one of P,W,M,D,C: " + task_name)
        if not (task[1] == "(" and task[-1] == ")"):
            raise ValueError("Wrong definition of task name ('%s'). Exit." % task)
        definition = task[2:-1]
        kwargs = {}
        if definition != "":
            options = definition.split(";")
            for j in range(len(options)):
                kvs = options[j].split("=")
                if kvs[0].lower() not in Config:
                    raise ValueError("Wrong parameter ('%s') of task name '%s'. Stop." % kvs[0], task_name)
                for k in range(len(kvs)):
                    kwargs[kvs[0].lower()] = kvs[1]
        actions_list.append((task_name, kwargs))


def work():
    for action in actions_list:
        print(datetime.datetime.now())
        print(" Start task " + action[0])
        func = actions_def[action[0]][0]
        kwargs = action[1]
        action_config_name = actions_def[action[0]][1]
        if len(action_config_name) > 0:
            action_config = get_configuration(parser, action_config_name)
            func(Config, action_config, kwargs)
        else:
            func(Config)


def parse_config_info(path):
    parser.read_file(open(path))
    for s in parser.sections():
        for opt in parser.items(s):
            Config[opt[0]] = opt[1]
    if not Config["home"]:
        Config["home"] = str(Path.home())
    if not Config["infofrom"]:
        Config["infofrom"] = "today"
    if Config["infofrom"] != "today":
        chk = Config["infofrom"].split()
        if len(chk) != 2 and not chk[1].startswith("day"):
            print ("Wrong value of 'infofrom' option. Exit.")
            return
        try:
            days = int(chk[0])
        except ValueError:
            print ("Wrong value of 'infofrom' option. Exit.")
            return
    test_path(Config, "reports_path", "Wrong path to the folder, containing reports. Exit.")
    test_path(Config, "actualpath",
              "Warning: wrong path to the folder containing original documents. It will not be possible to view them.")
    InfoCreator(Config)


