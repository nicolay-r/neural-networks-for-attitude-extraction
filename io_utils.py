import logging
from os import path
from os.path import dirname, join

from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from callback import CustomCallback
from model_io import CustomNeuralNetworkIO

from embeddings.rusvectores import RusvectoresEmbedding

from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.experiment.cv.default import SimpleCVFolding
from arekit.common.experiment.cv.sentence_based import SentenceBasedCVFolding
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.utils import create_dir_if_not_exists
from arekit.common.experiment.cv.doc_stat.rusentrel import RuSentRelDocStatGenerator
from arekit.common.experiment.data_io import DataIO
from arekit.contrib.networks.entities.str_fmt import StringSimpleMaskedEntityFormatter
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.io_utils import RuSentiFramesVersions

from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection

logger = logging.getLogger(__name__)


class RuSentRelBasedExperimentsIOUtils(DataIO):

    def __init__(self,
                 rusentrel_version=RuSentRelVersions.V11,
                 frames_version=RuSentiFramesVersions.V10,
                 model_states_dir=None,
                 labels_scaler=None):
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(frames_version, RuSentiFramesVersions))
        assert(isinstance(labels_scaler, BaseLabelScaler) or labels_scaler is None)

        super(RuSentRelBasedExperimentsIOUtils, self).__init__(
            labels_scale=ThreeLabelScaler() if labels_scaler is None else labels_scaler)

        self.__rusentrel_version = rusentrel_version
        self.__stemmer = MystemWrapper()
        self.__synonym_collection = RuSentRelSynonymsCollection.load_collection(
            stemmer=self.__stemmer,
            is_read_only=True)
        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter(self.__synonym_collection)
        self.__word_embedding = None
        self.__cv_folding_algorithm = self.__init_sentence_based_cv_folding_algorithm()

        self.__frames_collection = RuSentiFramesCollection.read_collection(version=frames_version)
        self.__unique_frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
            stemmer=self.__stemmer)

        self.__evaluator = TwoClassEvaluator(self.__synonym_collection)
        self.__callback = CustomCallback()

        self.__model_io = CustomNeuralNetworkIO(model_states_dir)

        self.__str_entity_formatter = self._init_str_entity_formatter()

    # region public properties

    @property
    def RuSentRelVersion(self):
        return self.__rusentrel_version

    @property
    def DistanceInTermsBetweenOpinionEndsBound(self):
        return 10

    @property
    def StringEntityFormatter(self):
        return self.__str_entity_formatter

    @property
    def Stemmer(self):
        return self.__stemmer

    @property
    def ModelIO(self):
        return self.__model_io

    @property
    def SynonymsCollection(self):
        return self.__synonym_collection

    @property
    def FramesCollection(self):
        return self.__frames_collection

    @property
    def FrameVariantCollection(self):
        return self.__unique_frame_variants

    @property
    def WordEmbedding(self):

        # Perform optional initialization
        if self.__word_embedding is None:
            self.__word_embedding = self.__create_word_embedding()

        return self.__word_embedding

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    @property
    def Evaluator(self):
        return self.__evaluator

    @property
    def CVFoldingAlgorithm(self):
        return self.__cv_folding_algorithm

    @property
    def Callback(self):
        return self.__callback

    # endregion

    # region private methods

    def _init_str_entity_formatter(self):
        return StringSimpleMaskedEntityFormatter()

    def __create_word_embedding(self):
        we_filepath = path.join(self.get_data_root(), u"w2v/news_rusvectores2.bin.gz")
        logger.info("Loading word embedding: {}".format(we_filepath))
        return RusvectoresEmbedding.from_word2vec_format(filepath=we_filepath,
                                                         binary=True)

    def __init_sentence_based_cv_folding_algorithm(self):
        return SentenceBasedCVFolding(
            docs_stat=RuSentRelDocStatGenerator(synonyms=self.__synonym_collection,
                                                version=self.__rusentrel_version),
            docs_stat_filepath=path.join(self.get_data_root(), u"docs_stat.txt"))

    def __init_simple_cv_folding_algoritm(self):
        return SimpleCVFolding()

    # endregion

    # region public methods

    def get_data_root(self):
        return path.join(dirname(__file__), u"data/")

    def get_experiments_dir(self):
        experiments_name = u'rusentrel'
        target_dir = join(self.get_data_root(), u"./{}/".format(experiments_name))
        create_dir_if_not_exists(target_dir)
        return target_dir

    def get_word_embedding_filepath(self):
        return path.join(self.get_data_root(), u"w2v/news_rusvectores2.bin.gz")

    # endregion
