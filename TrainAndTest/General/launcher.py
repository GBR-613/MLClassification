import os
import datetime
from pathlib import Path
from configparser import ConfigParser
from Preprocess.preprocess import Preprocessor
from WordEmbedding.vectors import Embedding
from Data.data import DataLoader
from Models.controller import ModelController
from Models.consolidation import Collector
from Utils.utils import get_configuration, get_absolute_path
from Info.creator import InfoCreator

Config = {}
parser = ConfigParser()
actions_list=[]

actions_def = {
    "P":(Preprocessor,"preprocess"),
    "W":(Embedding,"word_embedding"),
    "D":(DataLoader,"data"),
    "M":(ModelController,"model"),
    "C":(Collector,"")
}


def parse_config(path):
    parser.read_file(open(path))
    #try:
    sections = parser.sections()
    for i in range(len(sections)):
        options = parser.items(sections[i])
        #if sections[i] == "requests":
        #    if len(options) == 0 or not parser.has_option("requests", "request"):
        #        print ("Config file doesn't contain request for any process. Exit.")
        #        return
        for j in range(len(options)):
            Config[options[j][0]] = options[j][1]
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
    return parser
    #Config["error"] = False
    #grisha parseRequestAndLaunchPipe(parser, Config["request"])
    #except Error:
    #    raise Exception("Config file's parsing error. Exit.")


def parse_request(req):
    print ("=== Request " + Config["reqid"] + " ===")
    print(req)
    if len(req) == 0:
        req = (Config["request"]).strip().replace(" ", "")
    if len(req) == 0:
        raise ValueError("Request is not defined, nothing to do")
    tasks = req.split("|")
    for i in range(len(tasks)):
        task = tasks[i]
        process = task[0]
        if process not in actions_def.keys():
            raise ValueError("Wrong process: " + process)
        '''
        if not (process == "P" or process == "W" or process == "D" or process == "M" or process == "C"):
            print ("Request contains wrong name of process ('%s')."%(process))
            print ("It should be one of 'P' (preprocess), 'W' (word embedding), " +
                   "'D' (data definition), 'M' (model) or 'C' (consolidate results). Exit.")
            return
        '''
        if  not (task[1] == "(" and task[-1] == ")"):
            raise ValueError("Request contains wrong definition of process ('%s'). Exit."%(task))
        definition = task[2:-1]
        kwargs = {}
        if definition != "":
            options = definition.split(";")
            for j in range(len(options)):
                kvs = options[j].split("=")
                if kvs[0].lower() not in Config:
                    raise ValueError("Request contains wrong parameter ('%s') of process '%s'. Stop."%(kvs[0], process))
                for k in range(len(kvs)):
                    kwargs[kvs[0].lower()] = kvs[1]
        ''''           
        if process == "P":   #Preprocess
            DefConfig = get_configuration(parser, "preprocess")
            Preprocessor(Config, DefConfig, kwargs)
        elif process == "W":  #Word Embedding
            DefConfig = get_configuration(parser, "word_embedding")
            Embedding(Config, DefConfig, kwargs)
        elif process == "D":  #Load data
            DefConfig = get_configuration(parser, "data")
            DataLoader(Config, DefConfig, kwargs)
        elif process == "C": #Collector
            Collector(Config)
        else:    #Model
            DefConfig = get_configuration(parser, "model")
            ModelController(Config, DefConfig, kwargs)
        '''
        actions_list.append((process, kwargs))


def work():
    for i in range(len(actions_list)):
        action = actions_list[i]
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
    #try:
    sections = parser.sections()
    for i in range(len(sections)):
        options = parser.items(sections[i])
        for j in range(len(options)):
            Config[options[j][0]] = options[j][1]
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
    if len(Config["reports_path"]) == 0 or not os.path.isdir(get_absolute_path(Config, "reports_path")):
        print("Wrong path to the folder, containing reports. Exit.")
        return
    if len(Config["actualpath"]) == 0 or not os.path.isdir(get_absolute_path(Config, "actualpath")):
        print("Warning: wrong path to the folder, containing original documents.")
        print("It will not be possible to view this documents.")
    #except Error:
    #    print ("Config file's parsing error. Exit.")
    #    return
    InfoCreator(Config)


