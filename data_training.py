from arekit.common.experiment.data.training import TrainingData
from arekit.common.opinions.formatter import OpinionCollectionsFormatter
from arekit.processing.lemmatization.base import Stemmer
from callback import NeuralNetworkCustomEvaluationCallback


class RuSentRelTrainingData(TrainingData):

    def __init__(self, labels_scaler, stemmer, opinion_formatter, evaluator, callback):
        assert(isinstance(callback, NeuralNetworkCustomEvaluationCallback))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(opinion_formatter, OpinionCollectionsFormatter))

        super(RuSentRelTrainingData, self).__init__(labels_scaler, stemmer)

        self.__callback = callback
        self.__opinion_formatter = opinion_formatter
        self.__evaluator = evaluator

    @property
    def Evaluator(self):
        return self.__evaluator

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    @property
    def Callback(self):
        return self.__callback
