import logging

from arekit.common.embeddings.base import Embedding
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.opinions.formatter import OpinionCollectionsFormatter
from arekit.contrib.experiment_rusentrel.annot.algo import RuSentRelDefaultNeutralAnnotationAlgorithm
from arekit.contrib.experiment_rusentrel.annot.factory import ExperimentAnnotatorFactory
from arekit.contrib.experiment_rusentrel.frame_variants import ExperimentFrameVariantsCollection
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentPositiveLabel, ExperimentNegativeLabel
from arekit.contrib.networks.core.data.serializing import NetworkSerializationData
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.experiment_rusentrel.labels.formatters.rusentiframes import \
    ExperimentRuSentiFramesEffectLabelsFormatter, \
    ExperimentRuSentiFramesLabelsFormatter
from arekit.processing.lemmatization.base import Stemmer
from arekit.processing.pos.base import POSTagger

logger = logging.getLogger(__name__)


class RuSentRelExperimentSerializationData(NetworkSerializationData):

    def __init__(self,
                 labels_scaler,
                 stemmer,
                 pos_tagger,
                 embedding,
                 dist_in_terms_between_att_ends,
                 terms_per_context,
                 frames_version,
                 str_entity_formatter,
                 opinion_formatter,
                 rusentrel_version):
        assert(isinstance(embedding, Embedding))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(pos_tagger, POSTagger))
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(frames_version, RuSentiFramesVersions))
        assert(isinstance(str_entity_formatter, StringEntitiesFormatter))
        assert(isinstance(opinion_formatter, OpinionCollectionsFormatter))
        assert(isinstance(dist_in_terms_between_att_ends, int) or dist_in_terms_between_att_ends is None)
        assert(isinstance(terms_per_context, int))

        super(RuSentRelExperimentSerializationData, self).__init__(
            annot=ExperimentAnnotatorFactory.create(
                labels_count=labels_scaler.LabelsCount,
                create_algo=lambda: RuSentRelDefaultNeutralAnnotationAlgorithm(
                    dist_in_terms_bound=dist_in_terms_between_att_ends)),
            labels_scaler=labels_scaler,
            stemmer=stemmer)

        self.__pos_tagger = pos_tagger
        self.__terms_per_context = terms_per_context
        self.__rusentrel_version = rusentrel_version
        self.__str_entity_formatter = str_entity_formatter
        self.__word_embedding = embedding
        self.__opinion_formatter = opinion_formatter

        self.__frames_collection = RuSentiFramesCollection.read_collection(
            version=frames_version,
            labels_fmt=ExperimentRuSentiFramesLabelsFormatter(),
            effect_labels_fmt=ExperimentRuSentiFramesEffectLabelsFormatter())

        self.__unique_frame_variants = ExperimentFrameVariantsCollection(stemmer)

        # Filling collection.
        self.__unique_frame_variants.fill_from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants())

        # NOTE: Neutral and other labels.
        self.__frame_roles_label_scaler = ThreeLabelScaler()

        # NOTE: Considering only sentiment labels in results evaluation.
        self.__supported_collection_labels = {
            ExperimentPositiveLabel(),
            ExperimentNegativeLabel()
        }

    # region public properties

    @property
    def PosTagger(self):
        return self.__pos_tagger

    @property
    def StringEntityFormatter(self):
        return self.__str_entity_formatter

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

    @property
    def FrameRolesLabelScaler(self):
        return self.__frame_roles_label_scaler

    @property
    def SupportedCollectionLabels(self):
        return self.__supported_collection_labels

    # endregion
