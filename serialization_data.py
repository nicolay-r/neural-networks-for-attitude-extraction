import logging

from arekit.common.embeddings.base import Embedding
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.contrib.networks.core.data.serializing import NetworkSerializationData
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.networks.entities.str_fmt import StringSimpleMaskedEntityFormatter
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.io_utils import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from arekit.processing.lemmatization.mystem import MystemWrapper

logger = logging.getLogger(__name__)


class RuSentRelExperimentSerializationData(NetworkSerializationData):

    def __init__(self,
                 labels_scaler,
                 embedding,
                 terms_per_context,
                 frames_version,
                 str_entity_formatter,
                 rusentrel_version):
        assert(isinstance(embedding, Embedding))
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(frames_version, RuSentiFramesVersions))
        assert(isinstance(str_entity_formatter, StringEntitiesFormatter))
        assert(isinstance(terms_per_context, int))
        super(RuSentRelExperimentSerializationData, self).__init__(labels_scaler=labels_scaler)

        self.__terms_per_context = terms_per_context
        self.__rusentrel_version = rusentrel_version
        self.__str_entity_formatter = str_entity_formatter
        self.__stemmer = MystemWrapper()
        self.__word_embedding = embedding

        # Provide as a parameter.
        self.__synonym_collection = RuSentRelSynonymsCollection.load_collection(stemmer=self.__stemmer,
                                                                                is_read_only=True)

        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter(self.__synonym_collection)

        self.__frames_collection = RuSentiFramesCollection.read_collection(version=frames_version)
        self.__unique_frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
            stemmer=self.__stemmer)

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
        return self.__word_embedding

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

    # endregion
