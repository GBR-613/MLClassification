import os
import subprocess
import datetime
from subprocess import Popen, PIPE
from nltk.corpus import stopwords
from Utils.utils import get_formatted_date, get_abs_path, test_path


class TokensFromTagger:
    def __init__(self, Config):
        test_path(Config, "set_of_docs_lang_tokenization_lib_path",
                  "Wrong path to the tagger's jar. Tokenization can't be done")
        tagger_path = get_abs_path(Config, "set_of_docs_lang_tokenization_lib_path")
        inPath = Config["home"] + "/" + Config["source_path"]
        outPath = Config["home"] + "/" + Config["target_path"]
        stop_words = ",".join(list(stopwords.words('arabic'))) if Config["stop_words"] == "True" else ""
        ds = datetime.datetime.now()
        srv = subprocess.Popen('java -Xmx2g -jar ' + tagger_path + ' "' + inPath + '" "' +
                               outPath + '" "' + Config["exclude_positions"] + '" "'+ stop_words + '" "' +
                               Config["extra_words"] + '" "' + Config["normalization"] + '" "' +
                               Config["language_tokenization"] + '"',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        srv.wait()
        reply = srv.communicate()
        de = datetime.datetime.now()
        print(reply[0].decode())
        print("All process is done in %s" % (get_formatted_date(ds, de)))
