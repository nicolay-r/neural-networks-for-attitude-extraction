import argparse

from arekit.common.entities.formatters.factory import create_entity_formatter
from arekit.common.experiment.folding.types import FoldingType
from arekit.contrib.experiments.factory import create_experiment
from args.balance import UseBalancingArg
from args.dist_in_terms_between_ends import DistanceInTermsBetweenAttitudeEndsArg
from args.embedding import RusVectoresEmbeddingFilepathArg
from args.entity_fmt import EnitityFormatterTypesArg
from args.frames import RuSentiFramesVersionArg
from args.rusentrel import RuSentRelVersionArg
from args.stemmer import StemmerArg
from args.train_terms_per_context import TermsPerContextArg
from common import Common

from arekit.contrib.networks.run_serializer import NetworksExperimentInputSerializer
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

from args.cv_index import CvCountArg
from args.experiment import ExperimentTypeArg
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg
from data_serializing import RuSentRelExperimentSerializationData
from experiment_io import CustomNetworkExperimentIO

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data serializer (train/test) for RuSentRel-based experiments organization")

    # Providing arguments.
    RuAttitudesVersionArg.add_argument(parser)
    CvCountArg.add_argument(parser)
    ExperimentTypeArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)
    RusVectoresEmbeddingFilepathArg.add_argument(parser)
    TermsPerContextArg.add_argument(parser)
    RuSentiFramesVersionArg.add_argument(parser)
    RuSentRelVersionArg.add_argument(parser)
    EnitityFormatterTypesArg.add_argument(parser)
    StemmerArg.add_argument(parser)
    UseBalancingArg.add_argument(parser)
    DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser)

    parser.add_argument('--force',
                        dest='force',
                        type=bool,
                        const=True,
                        default=False,
                        nargs='?',
                        help='Perform forced data serialization')

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    exp_type = ExperimentTypeArg.read_argument(args)
    ra_version = RuAttitudesVersionArg.read_argument(args)
    cv_count = CvCountArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    embedding_filepath = RusVectoresEmbeddingFilepathArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    frames_version = RuSentiFramesVersionArg.read_argument(args)
    rusentrel_version = RuSentRelVersionArg.read_argument(args)
    entity_fmt = EnitityFormatterTypesArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)
    use_balancing = UseBalancingArg.read_argument(args)
    forced_serialization = args.force
    dist_in_terms_between_attitude_ends = DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)

    # Preparing necessary structures for further initializations.
    experiment_data = RuSentRelExperimentSerializationData(
        labels_scaler=Common.create_labels_scaler(labels_count),
        embedding=Common.load_rusvectores_word_embedding(embedding_filepath),
        terms_per_context=terms_per_context,
        frames_version=frames_version,
        rusentrel_version=rusentrel_version,
        str_entity_formatter=create_entity_formatter(entity_fmt),
        stemmer=stemmer,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends,
        opinion_formatter=Common.create_opinion_collection_formatter())

    extra_name_suffix = Common.create_exp_name_suffix(
        use_balancing=use_balancing,
        terms_per_context=terms_per_context,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    experiment = create_experiment(exp_type=exp_type,
                                   experiment_data=experiment_data,
                                   folding_type=FoldingType.Fixed if cv_count == 1 else FoldingType.CrossValidation,
                                   rusentrel_version=RuSentRelVersions.V11,
                                   ruattitudes_version=ra_version,
                                   experiment_io_type=CustomNetworkExperimentIO,
                                   extra_name_suffix=extra_name_suffix,
                                   load_ruattitude_docs=True)

    # Performing serialization process.
    serialization_engine = NetworksExperimentInputSerializer(experiment=experiment,
                                                             balance=use_balancing,
                                                             force_serialize=forced_serialization,
                                                             skip_folder_if_exists=True)

    serialization_engine.run()
