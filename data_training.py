from arekit.common.experiment.data.training import TrainingData
from arekit.common.opinions.formatter import OpinionCollectionsFormatter
from arekit.processing.lemmatization.base import Stemmer
from callback import NeuralNetworkCustomEvaluationCallback
from callback_eval import CallbackEvalF1NPU


class RuSentRelTrainingData(TrainingData):

    def __init__(self, labels_count, stemmer, opinion_formatter, evaluator, callback):
        assert(isinstance(labels_count, int))
        assert(isinstance(callback, NeuralNetworkCustomEvaluationCallback) or
               isinstance(callback, CallbackEvalF1NPU))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(opinion_formatter, OpinionCollectionsFormatter))

        super(RuSentRelTrainingData, self).__init__(stemmer=stemmer,
                                                    labels_count=labels_count)

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
