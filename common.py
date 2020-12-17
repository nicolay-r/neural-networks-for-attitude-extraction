import argparse
import logging

from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from embeddings.rusvectores import RusvectoresEmbedding


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Common:

    CV_NAME_PREFIX = u'cv_'

    @staticmethod
    def log_args(args):
        assert(isinstance(args, argparse.Namespace))
        logger.info("============")
        for arg in vars(args):
            logger.info(u"{}: {}".format(arg, getattr(args, arg)))
        logger.info("============")

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
    def create_exp_name_suffix(use_balancing, terms_per_context, dist_in_terms_between_att_ends):
        """ Provides an external parameters that assumes to be synchronized both
            by serialization and training experiment stages.
        """
        assert(isinstance(use_balancing, bool))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(dist_in_terms_between_att_ends, int) or dist_in_terms_between_att_ends is None)

        # You may provide your own parameters out there
        params = [
            u"balanced" if use_balancing else u"nobalance",
            u"tpc{}".format(terms_per_context)
        ]

        if dist_in_terms_between_att_ends is not None:
            params.append(u"dbe{}".format(dist_in_terms_between_att_ends))

        return u'-'.join(params)

    @staticmethod
    def create_labels_scaler(labels_count):
        assert(isinstance(labels_count, int))

        if labels_count == 2:
            return TwoLabelScaler()
        if labels_count == 3:
            return ThreeLabelScaler()

        raise NotImplementedError("Not supported")


