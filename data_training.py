from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.experiment.data.training import TrainingData
from arekit.common.opinions.formatter import OpinionCollectionsFormatter
from arekit.common.synonyms import SynonymsCollection
from arekit.processing.lemmatization.base import Stemmer
from callback import CustomCallback


class RuSentRelTrainingData(TrainingData):

    def __init__(self, labels_scaler, stemmer, synonyms, opinion_formatter, evaluator):
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(opinion_formatter, OpinionCollectionsFormatter))

        super(RuSentRelTrainingData, self).__init__(labels_scaler)

        self.__callback = CustomCallback()
        self.__stemmer = stemmer
        self.__synonym_collection = synonyms
        self.__opinion_formatter = opinion_formatter
        self.__evaluator = evaluator

    @property
    def Evaluator(self):
        return self.__evaluator

    @property
    def SynonymsCollection(self):
        return self.__synonym_collection

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    @property
    def Callback(self):
        return self.__callback
