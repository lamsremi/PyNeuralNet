"""
Keras model.
"""
import os

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers


class Model():
    """
    Keras ann model.
    """
    def __init__(self,
                 ):
        """
        Init the model instance.
        """
        self._model = None

    def predict(self, features):
        """
        Predict.
        """
        return self._model.predict(features)

    def fit(self, features, truth, input_dim, output_dim, batch_size, epochs):
        """
        Fit.
        """
        self._model = Sequential()
        self._model.add(
            Dense(10,
                  input_dim=input_dim,
                  activation="sigmoid",
                  kernel_initializer='ones',
                  bias_initializer='ones')
        )
        self._model.add(
            Dense(15,
                  activation="sigmoid",
                  kernel_initializer='ones',
                  bias_initializer='ones')
        )
        self._model.add(
            Dense(output_dim,
                  activation='linear',
                  kernel_initializer='ones',
                  bias_initializer='ones')
        )
        sgd = optimizers.SGD(lr=0.005,
                             decay=0,
                             momentum=0,
                             nesterov=False)
        self._model.compile(optimizer=sgd, loss="mean_absolute_error")
        self._model.fit(features, truth, batch_size=batch_size, epochs=epochs)

    def persist_parameters(self, model_version):
        """
        Persist parameters.
        """
        # Set folder of the param
        folder_path = "library/keras/params/{}/".format(model_version)
        # Create
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        self._model.save(folder_path + "/model.h5")

    def load_parameters(self, model_version):
        """
        Load parameters model.
        """
       # Set folder of the param
        folder_path = "library/keras/params/{}/".format(model_version)
        # Load the params
        self._model = load_model(folder_path + "/model.h5")
