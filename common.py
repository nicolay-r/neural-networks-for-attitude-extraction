import logging

from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from embeddings.rusvectores import RusvectoresEmbedding


logger = logging.getLogger(__name__)


class Common:

    CV_NAME_PREFIX = u'cv_'

    @staticmethod
    def load_rusvectores_word_embedding(filepath):
        logger.info("Loading word embedding: {}".format(filepath))
        return RusvectoresEmbedding.from_word2vec_format(filepath=filepath, binary=True)

    @staticmethod
    def create_opinion_collection_formatter():
        return RuSentRelOpinionCollectionFormatter()

    @staticmethod
    def create_full_model_name(folding_type, model_name):
        assert(isinstance(folding_type,  FoldingType))

        folding_prefix = u""
        if folding_type == FoldingType.CrossValidation:
            folding_prefix = Common.CV_NAME_PREFIX

        return u"{folding_type}{model_name}".format(
            folding_type=folding_prefix,
            model_name=model_name)

    @staticmethod
    def create_labels_scaler(labels_count):
        assert(isinstance(labels_count, int))

        if labels_count == 2:
            return TwoLabelScaler()
        if labels_count == 3:
            return ThreeLabelScaler()

        raise NotImplementedError("Not supported")


