import logging
from os import path
from os.path import dirname, join

from common import Common
from embeddings.rusvectores import RusvectoresEmbedding

from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.networks.core.data.serializing import NetworkSerializationData
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.networks.entities.str_fmt import StringSimpleMaskedEntityFormatter
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.io_utils import RuSentiFramesVersions

from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection

logger = logging.getLogger(__name__)


class RuSentRelSerializationData(NetworkSerializationData):

    def __init__(self,
                 labels_scaler,
                 frames_version=RuSentiFramesVersions.V10,
                 rusentrel_version=RuSentRelVersions.V11):
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(frames_version, RuSentiFramesVersions))
        super(RuSentRelSerializationData, self).__init__(labels_scaler)

        self.__rusentrel_version = rusentrel_version
        self.__stemmer = MystemWrapper()

        # Provide as a parameter.
        self.__synonym_collection = RuSentRelSynonymsCollection.load_collection(stemmer=self.__stemmer,
                                                                                is_read_only=True)

        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter(self.__synonym_collection)
        self.__cv_folding_algorithm = Common.create_folding_algorithm(synonyms=self.__synonym_collection,
                                                                      rusentrel_version=self.__rusentrel_version,
                                                                      data_dir=self.get_data_root())

        self.__frames_collection = RuSentiFramesCollection.read_collection(version=frames_version)
        self.__unique_frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
            stemmer=self.__stemmer)

        self.__str_entity_formatter = StringSimpleMaskedEntityFormatter()
        self.__word_embedding = None
        self.__sources_dir = None

    # region public properties

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
        if self.__word_embedding is None:
            self.__word_embedding = self.__create_word_embedding()
        return self.__word_embedding

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    @property
    def CVFoldingAlgorithm(self):
        return self.__cv_folding_algorithm

    @property
    def TermsPerContext(self):
        return 50

    # endregion

    # region private methods

    def __create_word_embedding(self):
        we_filepath = path.join(self.get_data_root(), u"w2v/news_rusvectores2.bin.gz")
        logger.info("Loading word embedding: {}".format(we_filepath))
        return RusvectoresEmbedding.from_word2vec_format(filepath=we_filepath,
                                                         binary=True)

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

    def get_experiment_sources_dir(self):
        src_dir = self.__sources_dir
        if self.__sources_dir is None:
            # Considering a source dir by default.
            src_dir = join(self.get_data_root())
        return src_dir

    # endregion
