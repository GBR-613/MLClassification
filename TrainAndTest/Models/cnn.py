import datetime
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.layers import Convolution1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import AlphaDropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Utils.utils import arabic_charset, get_absolute_path, show_time

from keras import backend as K
import tensorflow as tf


class CNNModel(BaseModel):
    def __init__(self, Config):
        super().__init__(Config)
        try:
            self.validation_data_size = float(Config["validation_data_size"])
        except ValueError:
            self.validation_data_size = 0
        if self.validation_data_size <= 0 or self.validation_data_size >= 1:
            raise ValueError("Wrong size of validation data set. Stop.")
        self.addValSet = True
        self.handleType = "charVectors"
        self.save_intermediate_results = Config["save_intermediate_results"] == "True"
        self.useProbabilities = True
        if Config["type_of_execution"] != "crossvalidation":
            self.prepareData()
        self.launchProcess()

    def prepareData(self):
        print("Start data preparation...")
        dp = DataPreparation(self, self.addValSet)
        dp.getCharVectors()

    def createModel(self):
        embeddingSize = 128
        maxSeqLength = self.Config["max_chars_seq_len"]
        convLayersData = [[256, 10], [256, 7], [256, 5], [256, 3]]
        dropout_p = 0.1
        optimizer = 'adam'
        inputs = Input(shape=(maxSeqLength,), dtype='int64')
        x = Embedding(len(arabic_charset()) + 1, embeddingSize, input_length=maxSeqLength)(inputs)
        convolution_output = []
        for num_filters, filter_width in convLayersData:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_width,
                                 activation='tanh')(x)
            pool = GlobalMaxPooling1D()(conv)
            convolution_output.append(pool)
        x = Concatenate()(convolution_output)
        x = Dense(1024, activation='selu', kernel_initializer='lecun_normal')(x)
        x = AlphaDropout(dropout_p)(x)
        x = Dense(1024, activation='selu', kernel_initializer='lecun_normal')(x)
        x = AlphaDropout(dropout_p)(x)
        predictions = Dense(len(self.Config["predefined_categories"]), activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def loadModel(self):
        self.model = self.loadNNModel()

    def trainModel(self):
        """
        cf = tf.ConfigProto(inter_op_parallelism_threads=5)
        session = tf.Session(config=cf)
        K.set_session(session)
        """
        self.trainNNModel()

    def testModel(self):
        self.testNNModel()

    def saveAdditions(self):
        self.resources["handleType"] = "charVectors"
