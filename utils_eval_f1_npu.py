import argparse
from os.path import exists

from arekit.common.entities.formatters.types import EntityFormattersService
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
from arekit.contrib.experiments.types import ExperimentTypes
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.enum_input_types import ModelInputTypeService
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersionsService
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersionsService
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from args.rusentrel import RuSentRelVersionArg
from args.stemmer import StemmerArg
from args.terms_per_context import TermsPerContextArg
from args.train.model_name_tag import ModelNameTagArg
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
        print self._experiment.DocumentOperations
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

    parser = argparse.ArgumentParser(description='F1-pnu evaluator')

    parser.add_argument('--max-epochs',
                        dest="max_epochs",
                        type=int,
                        default=200,
                        nargs=1,
                        help="Labels count in an output classifier")

    # Parsing arguments.
    args = parser.parse_args()

    labels_count = 3
    balanced_input = True
    max_epochs_count = args.max_epochs[0]
    terms_per_context = TermsPerContextArg.default
    rusentrel_version = RuSentRelVersionArg.default
    stemmer = StemmerArg.supported[StemmerArg.default]
    labels_scaler = create_labels_scaler(labels_count)
    eval_mode = EvaluationModes.Extraction
    dist_in_terms_between_attitude_ends = None

    grid = {
        u"foldings": [FoldingType.Fixed, FoldingType.CrossValidation],
        u"exp_types": [ExperimentTypes.RuSentRel,
                       ExperimentTypes.RuSentRelWithRuAttitudes],
        u"entity_fmts": [EntityFormattersService.get_type_by_name(ent_fmt)
                         for ent_fmt in EntityFormattersService.iter_supported_names()],
        u"ra_names": [RuAttitudesVersionsService.find_by_name(ra_name)
                      for ra_name in RuAttitudesVersionsService.iter_supported_names()],
        u"input_types": [ModelInputTypeService.get_type_by_name(input_type)
                         for input_type in ModelInputTypeService.iter_supported_names()],
        u"frames_versions": [RuSentiFramesVersionsService.get_type_by_name(frames_version)
                             for frames_version in RuSentiFramesVersionsService.iter_supported_names()],
        u'model_names': Common.default_results_considered_model_names_list(),
        u'balancing': [True, False],
        u'model_name_tags': [ModelNameTagArg.NO_TAG]
    }

    def __run():

        # Setup default evaluator.
        evaluator = ThreeClassEvaluator(DataType.Test)

        experiment_data = RuSentRelTrainingData(
            labels_scaler=labels_scaler,
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
                                       folding_type=folding_type,
                                       rusentrel_version=rusentrel_version,
                                       experiment_io_type=CustomNetworkExperimentIO,
                                       ruattitudes_version=ra_version,
                                       load_ruattitude_docs=False,
                                       extra_name_suffix=extra_name_suffix)

        full_model_name = Common.create_full_model_name(folding_type=folding_type,
                                                        model_name=model_name,
                                                        input_type=model_input_type)

        model_io = NeuralNetworkModelIO(full_model_name=full_model_name,
                                        target_dir=experiment.ExperimentIO.get_target_dir(),
                                        # From this depends on whether we have a specific dir or not.
                                        source_dir=None if model_name_tag == ModelNameTagArg.NO_TAG else object(),
                                        model_name_tag=model_name_tag)

        # Setup model io.
        experiment_data.set_model_io(model_io)

        # Check dir existance in advance.
        if not exists(model_io.get_model_dir()):
            return

        engine = ExperimentF1PNUEvaluator(experiment=experiment,
                                          data_type=DataType.Test,
                                          max_epochs_count=max_epochs_count)

        # Starting evaluation process.
        engine.run()

    for folding_type in grid[u"foldings"]:
        for exp_type in grid[u'exp_types']:
            for entity_formatter_type in grid[u'entity_fmts']:
                for model_name in grid[u'model_names']:
                    for balanced_input in grid[u'balancing']:
                        for ra_version in grid[u'ra_names']:
                            for model_input_type in grid[u'input_types']:
                                for frames_version in grid[u'frames_versions']:
                                    for model_name_tag in grid[u'model_name_tags']:
                                        __run()