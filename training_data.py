from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.experiment.data.training import TrainingData
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from arekit.processing.lemmatization.mystem import MystemWrapper
from callback import CustomCallback


class RuSentRelTrainingData(TrainingData):

    def __init__(self, labels_scaler):
        super(RuSentRelTrainingData, self).__init__(labels_scaler)

        self.__callback = CustomCallback()
        self.__stemmer = MystemWrapper()
        self.__synonym_collection = RuSentRelSynonymsCollection.load_collection(
            stemmer=self.__stemmer,
            is_read_only=True)
        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter(self.__synonym_collection)
        self.__evaluator = TwoClassEvaluator(self.__synonym_collection)

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
