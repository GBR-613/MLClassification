import os
import subprocess
import datetime
from subprocess import Popen, PIPE
from nltk.corpus import stopwords
from Utils.utils import get_formatted_date, get_abs_path, updateParams


def job_preprocessor(Config, DefConfig, kwargs):
    worker = Preprocessor(Config, DefConfig, kwargs)
    worker.run()


class Preprocessor:
    def __init__(self, Config, DefConfig, kwargs):
        print("=== Preprocessing ===")
        updateParams(Config, DefConfig, kwargs)
        self.Config = Config
        self.DefConfig = DefConfig
        #self.process(Config)

    def run(self):
        lib_path = get_abs_path(self.Config, "set_of_docs_lang_tokenization_lib_path")
        print("GRISHA use set_of_docs_lang_tokenization")
        if not lib_path or not os.path.exists(lib_path):
            raise ValueError("Wrong path to the tagger's jar. Tokenization can't be done")
        in_path = self.Config["home"] + "/" + self.Config["source_path"]
        if not self.Config["source_path"] or self.Config["source_path"] == self.Config["target_path"]:
            raise ValueError("Wrong source/target path(s). Tokenization can't be done.")
        out_path = self.Config["home"] + "/" + self.Config["target_path"]
        stop_words = ""
        stop_words = ",".join(list(stopwords.words('arabic'))) if self.Config["stop_words"] == "True" else ""
        ds = datetime.datetime.now()
        srv = subprocess.Popen('java -Xmx2g -jar ' + lib_path + ' "' + in_path + '" "' +
                               out_path + '" "' + self.Config["exclude_positions"] + '" "'+ stop_words + '" "' +
                               self.Config["extra_words"] + '" "' + self.Config["normalization"] + '" "' +
                               self.Config["language_tokenization"] + '"',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        srv.wait()
        reply = srv.communicate()
        de = datetime.datetime.now()
        print(reply[0].decode())
        print("All process is done in %s" % (get_formatted_date(ds, de)))
