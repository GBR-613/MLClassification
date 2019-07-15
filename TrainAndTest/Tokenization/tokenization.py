from Tokenization.server import TokensFromServer
from Tokenization.tagger import TokensFromTagger
from Utils.utils import updateParams


class Tokenizer:
    def __init__(self, Config, DefConfig, kwargs):
        print ("=== Tokenization ===")
        updateParams(Config, DefConfig, kwargs)
        self.Config = Config
        self.DefConfig = DefConfig
        #if Config["language_tokenization"] != "True":
        #    return
        if not Config["source_path"] or Config["source_path"] == Config["target_path"]:
            print ("Wrong source/target path(s). Tokenization can't be done.")
            Config["error"] = True
            return
        if Config["typetoks"] == "server":
            TokensFromServer(Config)
        elif Config["typetoks"] == "tagger":
            TokensFromTagger(Config)
        else:
            print ("Wrong tokenization type. Tokenization can't be done.")
