import argparse
from os.path import exists

from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.evaluators.three_class import ThreeClassEvaluator
from arekit.common.experiment.data.training import TrainingData
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine.cv_based import ExperimentEngine
from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.scales.factory import create_labels_scaler
from arekit.contrib.bert.callback import Callback
from arekit.contrib.experiments.factory import create_experiment
from arekit.contrib.experiments.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from args.balance import UseBalancingArg
from args.cv_index import CvCountArg
from args.dist_in_terms_between_ends import DistanceInTermsBetweenAttitudeEndsArg
from args.entity_fmt import EnitityFormatterTypesArg
from args.experiment import ExperimentTypeArg
from args.frames import RuSentiFramesVersionArg
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg
from args.rusentrel import RuSentRelVersionArg
from args.stemmer import StemmerArg
from args.terms_per_context import TermsPerContextArg
from args.train.model_input_type import ModelInputTypeArg
from args.train.model_name import ModelNameArg
from callback_eval import CallbackEvalF1NPU
from callback_eval_func import calculate_results
from common import Common
from data_training import RuSentRelTrainingData
from experiment_io import CustomNetworkExperimentIO


class ExperimentF1PNUEvaluator(ExperimentEngine):

    def __init__(self, experiment, data_type, max_epochs_count):
        assert(isinstance(max_epochs_count, int))

        super(ExperimentF1PNUEvaluator, self).__init__(experiment)

        self.__data_type = data_type
        self.__max_epochs_count = max_epochs_count

        # NOTE:
        # This is the only limitation: we provide synonyms collection.
        self.__synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
            stemmer=self._experiment.DataIO.Stemmer,
            version=rusentrel_version)

    def __get_target_dir(self):
        return self._experiment.ExperimentIO.get_target_dir()

    def _handle_iteration(self, iter_index):
        exp_data = self._experiment.DataIO
        assert(isinstance(exp_data, TrainingData))

        # Setup callback.
        callback = exp_data.Callback
        assert(isinstance(callback, Callback))
        callback.set_iter_index(iter_index)
        cmp_doc_ids_set = set(self._experiment.DocumentOperations.iter_doc_ids_to_compare())

        exp_io = self._experiment.ExperimentIO
        assert(isinstance(exp_io, NetworkIOUtils))

        with callback:
            for epoch_index in range(self.__max_epochs_count):

                target_file = exp_io.get_output_model_results_filepath(
                    data_type=self.__data_type,
                    epoch_index=epoch_index)

                if not exists(target_file):
                    continue

                print "Found:", target_file

                # Calculate results.
                calculate_results(
                    doc_ids=cmp_doc_ids_set,
                    synonyms=self.__synonyms,
                    evaluator=exp_data.Evaluator,
                    iter_etalon_opins_by_doc_id_func=lambda doc_id:
                        self._experiment.OpinionOperations.try_read_neutrally_annotated_opinion_collection(
                            doc_id=doc_id,
                            data_type=self.__data_type),
                    iter_result_opins_by_doc_id_func=lambda doc_id:
                        self._experiment.OpinionOperations.read_result_opinion_collection(
                            data_type=self.__data_type,
                            doc_id=doc_id,
                            epoch_index=epoch_index))

                # evaluate
                result = self._experiment.evaluate(data_type=self.__data_type,
                                                   epoch_index=epoch_index)
                result.calculate()

                # saving results.
                callback.write_results(result=result,
                                       data_type=self.__data_type,
                                       epoch_index=epoch_index)

    def _before_running(self):
        # Providing a root dir for logging.
        callback = self._experiment.DataIO.Callback
        callback.set_log_dir(self.__get_target_dir())


if __name__ == "__main__":

    # TODO. Make this as a test.

    parser = argparse.ArgumentParser(description='*.tsv results based evaluator')

    ExperimentTypeArg.add_argument(parser)
    CvCountArg.add_argument(parser)
    RuSentRelVersionArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)
    RuAttitudesVersionArg.add_argument(parser)
    UseBalancingArg.add_argument(parser)
    TermsPerContextArg.add_argument(parser)
    DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser)
    StemmerArg.add_argument(parser)
    RuSentiFramesVersionArg.add_argument(parser)
    EnitityFormatterTypesArg.add_argument(parser)
    ModelInputTypeArg.add_argument(parser)
    ModelNameArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    labels_count = 3
    balanced_input = True
    max_epochs_count = 100
    stemmer = StemmerArg.read_argument(args)
    exp_type = ExperimentTypeArg.read_argument(args)
    cv_count = CvCountArg.read_argument(args)
    rusentrel_version = RuSentRelVersionArg.read_argument(args)
    entity_formatter_type = EnitityFormatterTypesArg.read_argument(args)
    labels_scaler = create_labels_scaler(labels_count)
    model_name = ModelNameArg.read_argument(args)
    ra_version = RuAttitudesVersionArg.read_argument(args)
    folding_type = FoldingType.Fixed if cv_count == 1 else FoldingType.CrossValidation
    balance_samples = UseBalancingArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    dist_in_terms_between_attitude_ends = DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)
    frames_version = RuSentiFramesVersionArg.read_argument(args)
    model_input_type = ModelInputTypeArg.read_argument(args)
    eval_mode = EvaluationModes.Extraction

    full_model_name = Common.create_full_model_name(folding_type=folding_type,
                                                    model_name=model_name,
                                                    input_type=model_input_type)

    # Setup default evaluator.
    evaluator = ThreeClassEvaluator(DataType.Test)

    experiment_data = RuSentRelTrainingData(
        labels_scaler=create_labels_scaler(labels_count),
        stemmer=stemmer,
        evaluator=evaluator,
        opinion_formatter=RuSentRelOpinionCollectionFormatter(),
        callback=CallbackEvalF1NPU(DataType.Test))

    extra_name_suffix = Common.create_exp_name_suffix(
        use_balancing=balanced_input,
        terms_per_context=terms_per_context,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    # Composing experiment.
    experiment = create_experiment(exp_type=exp_type,
                                   experiment_data=experiment_data,
                                   folding_type=FoldingType.Fixed if cv_count == 1 else FoldingType.CrossValidation,
                                   rusentrel_version=rusentrel_version,
                                   experiment_io_type=CustomNetworkExperimentIO,
                                   ruattitudes_version=ra_version,
                                   load_ruattitude_docs=False,
                                   extra_name_suffix=extra_name_suffix)

    engine = ExperimentF1PNUEvaluator(experiment=experiment,
                                      data_type=DataType.Test,
                                      max_epochs_count=max_epochs_count)

    # Starting evaluation process.
    engine.run()