from os.path import join, dirname
from arekit.common.experiment.data.training import TrainingData
from callback import CustomCallback


class RuSentRelTrainingData(TrainingData):

    def __init__(self, labels_scaler):
        super(RuSentRelTrainingData, self).__init__(labels_scaler)
        
        self.__sources_dir = None
        self.__results_dir = None
        self.__callback = CustomCallback()

    @property
    def Evaluator(self):
        return self.__evaluator

    @property
    def SynonymsCollection(self):
        pass

    @property
    def OpinionFormatter(self):
        pass

    @property
    def CVFoldingAlgorithm(self):
        pass

    @property
    def Callback(self):
        return self.__callback

    def get_data_root(self):
        return join(dirname(__file__), u"data/")

    def get_experiment_sources_dir(self):
        src_dir = self.__sources_dir
        if self.__sources_dir is None:
            # Considering a source dir by default.
            src_dir = join(self.get_data_root())
        return src_dir

    def get_experiment_results_dir(self):
        if self.__results_dir is None:
            # Considering the same as a source dir
            return self.get_experiment_sources_dir()
        return self.__results_dir
