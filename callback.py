import logging
import datetime
from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.utils import create_dir_if_not_exists

from arekit.contrib.networks.core.callback.base import Callback
from arekit.contrib.networks.core.callback.utils_hidden_states import save_model_hidden_values
from arekit.contrib.networks.core.callback.utils_model_eval import evaluate_model
from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from callback_log_utils import create_verbose_eval_results_msg, create_overall_eval_results_msg, write_config_setups

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuralNetworkCustomEvaluationCallback(Callback):

    __costs_window = 5

    __log_train_filename_template = u"cb_train_{iter}_{dtype}.log"
    __log_eval_filename_template = u"cb_eval_{iter}_{dtype}.log"
    __log_eval_verbose_filename = u"cb_eval_verbose_{iter}_{dtype}.log"

    def __init__(self, do_eval,
                 cancellation_acc_bound,
                 cancellation_f1_train_bound):
        assert(isinstance(do_eval, bool))

        super(NeuralNetworkCustomEvaluationCallback, self).__init__()

        self.__test_on_epochs = None
        self.__experiment = None
        self.__model = None
        self.__log_dir = None
        self.__do_eval = do_eval
        self.__costs_history = None
        self.__key_stop_training_by_cost = None

        self.__train_log_files = {}
        self.__eval_log_files = {}
        self.__eval_verbose_log_files = {}

        self.__key_save_hidden_parameters = True

        self.__cancellation_acc_bound = cancellation_acc_bound
        self.__cancellation_f1_train_bound = cancellation_f1_train_bound

    # region properties

    @property
    def Epochs(self):
        return max(self.__test_on_epochs)

    # endregion

    # region public methods

    def set_experiment(self, experiment):
        assert(isinstance(experiment, BaseExperiment))
        self.__experiment = experiment

    def set_test_on_epochs(self, value):
        assert(isinstance(value, list))
        self.__test_on_epochs = value

    def set_log_dir(self, log_dir):
        assert(isinstance(log_dir, unicode))
        self.__log_dir = log_dir

    def set_key_save_hidden_parameters(self, value):
        assert(isinstance(value, bool))
        self.__key_save_hidden_parameters = value

    def on_initialized(self, model):
        assert(isinstance(model, BaseTensorflowModel))
        self.__model = model
        write_config_setups(config=model.Config,
                            out_filepath=join(self.__log_dir, u"model_config.txt"))

    def on_experiment_iteration_begin(self):
        self.__costs_history = []

    def set_key_stop_training_by_cost(self, value):
        assert(isinstance(value, bool))
        self.__key_stop_training_by_cost = value

    # endregion

    # region private methods

    def __check_costs_still_improving(self, avg_cost):
        history_len = len(self.__costs_history)
        if history_len <= self.__costs_window:
            return True
        return avg_cost < min(self.__costs_history[:history_len - self.__costs_window])

    def __is_cancel_needed(self, result_train, avg_fit_cost):
        """ This method related to the main algorithm that defines
            whether there is a need to stop training process or not.
        """
        msg = None
        cancel = False

        f1_train = result_train.get_result_by_metric(result_train.C_F1)
        if f1_train >= self.__cancellation_f1_train_bound:
            msg = u"Stop Training Process: F1-train ({f1_actual}) > {f1_bound}".format(
                f1_actual=round(f1_train, 3),
                f1_bound=self.__cancellation_f1_train_bound)
            cancel = True

        if self.__key_stop_training_by_cost:
            if not self.__check_costs_still_improving(avg_fit_cost):
                msg = u"Stop Training Process: cost becomes greater than min value {} epochs ago.".format(
                    self.__costs_window)
                cancel = True

        if msg is not None:
            logger.info(msg)

        return cancel

    def __test_and_log_results(self, operation_cancel, epoch_index, avg_fit_cost):
        assert(isinstance(operation_cancel, OperationCancellation))

        # We use the latter temporary. Maybe it might be in a way better to refactor this aspect.
        # For now such formatter could not be taken from the related experiment.
        eval_label_formatter = RuSentRelLabelsFormatter()

        result = {}

        for data_type in self.__experiment.DocumentOperations.DataFolding.iter_supported_data_types():
            assert(isinstance(data_type, DataType))
            result[data_type] = evaluate_model(
                experiment=self.__experiment,
                save_hidden_params=self.__key_save_hidden_parameters,
                data_type=data_type,
                epoch_index=epoch_index,
                model=self.__model,
                labels_formatter=eval_label_formatter,
                log_dir=self.__log_dir)

        # Saving the obtained results.
        for data_type, eval_result in result.iteritems():
            self.__save_evaluation_results(result=eval_result,
                                           data_type=data_type,
                                           epoch_index=epoch_index)

        # Check whether there is a need to stop training process.
        if self.__is_cancel_needed(result_train=result[DataType.Train], avg_fit_cost=avg_fit_cost):
            operation_cancel.Cancel()

        self.__costs_history.append(avg_fit_cost)

    def __save_evaluation_results(self, result, data_type, epoch_index):
        eval_verbose_msg = create_verbose_eval_results_msg(eval_result=result,
                                                           data_type=data_type,
                                                           epoch_index=epoch_index)

        eval_msg = create_overall_eval_results_msg(eval_result=result,
                                                   data_type=data_type,
                                                   epoch_index=epoch_index)

        # Writing evaluation logging results.
        logger.info(eval_msg)

        # Separate logging information by files.
        self.__eval_log_files[data_type].write(u"{}\n".format(eval_msg))
        self.__eval_verbose_log_files[data_type].write(u"{}\n".format(eval_verbose_msg))

    # endregion

    def on_fit_started(self, operation_cancel):
        if not self.__do_eval:
            return

        # Providing information into main logger.
        message = u"{}: Initial Evaluation:".format(str(datetime.datetime.now()))
        logger.info(message)

        self.__test_and_log_results(operation_cancel=operation_cancel,
                                    epoch_index=0,
                                    avg_fit_cost=-1)

    def on_epoch_finished(self, avg_fit_cost, avg_fit_acc, epoch_index, operation_cancel):
        assert(isinstance(avg_fit_cost, float))
        assert(isinstance(avg_fit_acc, float))
        assert(isinstance(epoch_index, int))
        assert(isinstance(operation_cancel, OperationCancellation))

        message = u"{}: Epoch: {}: avg_fit_cost: {:.3f}, avg_fit_acc: {:.3f}".format(
            str(datetime.datetime.now()),
            epoch_index,
            avg_fit_cost,
            avg_fit_acc)

        # Providing information into main logger.
        logger.info(message)

        # Duplicate the related information in separate log file.
        self.__train_log_files[DataType.Train].write(u"{}\n".format(message))

        if avg_fit_acc >= self.__cancellation_acc_bound:
            logger.info(u"Stop feeding process: avg_fit_acc > {}".format(self.__cancellation_acc_bound))
            operation_cancel.Cancel()

        # Deciding whether there is a need in evaluation process organization.
        is_need_eval = epoch_index in self.__test_on_epochs or operation_cancel.IsCancelled

        # Performing evaluation process (optionally).
        if self.__do_eval and is_need_eval:
            self.__test_and_log_results(operation_cancel=operation_cancel,
                                        epoch_index=epoch_index,
                                        avg_fit_cost=avg_fit_cost)

        # Check, whether there is a need to proceed with keeping hidden states or not.
        if (epoch_index not in self.__test_on_epochs) and (not operation_cancel.IsCancelled):
            return

        # Saving model hidden values using the related numpy utils.
        save_model_hidden_values(log_dir=self.__log_dir,
                                 epoch_index=epoch_index,
                                 save_hidden_parameters=self.__key_save_hidden_parameters,
                                 model=self.__model)

    # region base methods

    def __enter__(self):
        assert(self.__log_dir is not None)

        iter_index = str(self.__experiment.DocumentOperations.DataFolding.IterationIndex)

        for d_type in self.__experiment.DocumentOperations.DataFolding.iter_supported_data_types():

            train_log_filepath = join(self.__log_dir, self.__log_train_filename_template.format(iter=iter_index,
                                                                                                dtype=d_type))
            eval_log_filepath = join(self.__log_dir, self.__log_eval_filename_template.format(iter=iter_index,
                                                                                              dtype=d_type))
            eval_verbose_log_filepath = join(self.__log_dir, self.__log_eval_verbose_filename.format(iter=iter_index,
                                                                                                     dtype=d_type))

            create_dir_if_not_exists(train_log_filepath)
            create_dir_if_not_exists(eval_log_filepath)
            create_dir_if_not_exists(eval_verbose_log_filepath)

            self.__train_log_files[d_type] = open(train_log_filepath, u"w", buffering=0)
            self.__eval_log_files[d_type] = open(eval_log_filepath, u"w", buffering=0)
            self.__eval_verbose_log_files[d_type] = open(eval_verbose_log_filepath, u"w", buffering=0)

    def __exit__(self, exc_type, exc_val, exc_tb):

        for d_type in self.__experiment.DocumentOperations.DataFolding.iter_supported_data_types():

            self.__train_log_file = self.__eval_log_files[d_type]
            self.__eval_log_file = self.__eval_log_files[d_type]
            self.__eval_verbose_log_file = self.__eval_verbose_log_files[d_type]

            if self.__train_log_file is not None:
                self.__train_log_file.close()

            if self.__eval_log_file is not None:
                self.__eval_log_file.close()

            if self.__eval_verbose_log_file is not None:
                self.__eval_verbose_log_file.close()

    # endregion