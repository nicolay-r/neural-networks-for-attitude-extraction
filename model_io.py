import os
from arekit.contrib.networks.core.nn_io import NeuralNetworkModelIO


class CustomNeuralNetworkIO(NeuralNetworkModelIO):

    def __init__(self, model_states_dir=None):
        assert(isinstance(model_states_dir, str) or model_states_dir is None)
        self.__model_root = None
        self.__model_name = None
        self.__model_states_dir = model_states_dir

    @property
    def ModelRoot(self):
        return self.__model_root

    def set_model_root(self, value):
        assert(isinstance(value, unicode))
        self.__model_root = value

    def set_model_name(self, value):
        assert(isinstance(value, unicode))
        self.__model_name = value

    @property
    def ModelSavePathPrefix(self):
        return os.path.join(self.__model_states_dir, u'{}'.format(self.__model_name))


