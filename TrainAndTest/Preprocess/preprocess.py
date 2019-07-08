import os
import subprocess
import datetime
from subprocess import Popen, PIPE
from nltk.corpus import stopwords
from Utils.utils import show_time
from Utils.utils import get_absolute_path, updateParams


class Preprocessor:
    def __init__(self, Config, DefConfig, kwargs):
        print("=== Preprocessing ===")
        updateParams(Config, DefConfig, kwargs)
        self.Config = Config
        self.DefConfig = DefConfig
        self.process(Config)

    def process(self, Config):
        lib_path = get_absolute_path(Config, "set_of_docs_lang_tokenization_lib_path")
        print("GRISHA use set_of_docs_lang_tokenization")
        if len(lib_path) == 0 or not os.path.exists(lib_path):
            raise ValueError("Wrong path to the tagger's jar. Tokenization can't be done")
        in_path = Config["home"] + "/" + Config["sourcepath"]
        if len(Config["sourcepath"]) == 0 or Config["sourcepath"] == Config["targetpath"]:
            raise ValueError("Wrong source/target path(s). Tokenization can't be done.")
        out_path = Config["home"] + "/" + Config["targetpath"]
        stop_words = ""
        if Config["stopwords"] == "True":
            sWords = list(stopwords.words('arabic'))
            for i in range(len(sWords)):
                if i > 0:
                    stop_words += ","
                stop_words += sWords[i]
        ds = datetime.datetime.now()
        srv = subprocess.Popen('java -Xmx2g -jar ' + lib_path + ' "' + in_path +  '" "'  +
                               out_path + '" "' + Config["exclude_positions"] + '" "'+ stop_words + '" "' +
                               Config["extrawords"] + '" "' + Config["normalization"] + '" "' +
                               Config["language_tokenization"] + '"',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        srv.wait()
        reply = srv.communicate()
        de = datetime.datetime.now()
        print(reply[0].decode())
        print("All process is done in %s" % (show_time(ds, de)))
