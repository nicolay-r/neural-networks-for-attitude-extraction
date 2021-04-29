import logging
import datetime
from collections import OrderedDict
from itertools import chain
from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.utils import create_dir_if_not_exists

from arekit.contrib.networks.core.callback.base import Callback
from arekit.contrib.networks.core.callback.utils_hidden_states import save_model_hidden_values
from arekit.contrib.networks.core.callback.utils_model_eval import evaluate_model
from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.experiment_rusentrel.evaluation.results.two_class import TwoClassEvalResult

from callback_log_cfg import write_config_setups
from callback_log_exp import create_experiment_eval_msgs
from callback_log_iter import create_iteration_short_eval_msg, create_iteration_verbose_eval_msg
from callback_log_training import get_message
from common import Common

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuralNetworkCustomEvaluationCallback(Callback):

    __costs_window = 5

    __log_eval_iter_verbose_filename = u"cb_eval_verbose_{iter}_{dtype}.log"

    def __init__(self, do_eval,
                 label_calc_mode,
                 train_acc_limit,
                 train_f1_limit):
        assert(isinstance(train_acc_limit, float) or train_acc_limit is None)
        assert(isinstance(train_f1_limit, float) or train_f1_limit is None)
        assert(isinstance(label_calc_mode, LabelCalculationMode))
        assert(isinstance(do_eval, bool))

        super(NeuralNetworkCustomEvaluationCallback, self).__init__()

        self.__experiment = None
        self.__model = None
        self.__label_calc_mode = label_calc_mode

        self.__test_results_exp_history = OrderedDict()
        self.__eval_on_epochs = None
        self.__log_dir = None
        self.__do_eval = do_eval
        self.__key_stop_training_by_cost = None

        self.__train_iter_log_files = {}
        self.__eval_iter_log_files = {}
        self.__eval_iter_verbose_log_files = {}

        self.__key_save_hidden_parameters = True

        # Training cancellation related parameters.
        # TODO. Assumes to be moved into a separated class with the related logic.
        self.__train_iteration_costs_history = None
        self.__train_acc_limit = train_acc_limit
        self.__train_f1_limit = train_f1_limit

    # region properties

    @property
    def Epochs(self):
        return max(self.__eval_on_epochs) + 1

    # endregion

    # region public methods

    def set_experiment(self, experiment):
        assert(isinstance(experiment, BaseExperiment))
        self.__experiment = experiment

    def set_eval_on_epochs(self, value):
        assert(isinstance(value, list))
        self.__eval_on_epochs = value

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
                            out_filepath=join(self.__log_dir, Common.model_config_name))

    def on_experiment_iteration_begin(self):
        self.__train_iteration_costs_history = []

    def set_key_stop_training_by_cost(self, value):
        assert(isinstance(value, bool))
        self.__key_stop_training_by_cost = value

    def on_experiment_finished(self):
        """ Providing results aggregation across all the experiment iterations.
        """

        # Considering that we perform evaluations per every iteration.
        if not self.__do_eval:
            return

        # Opening the related file.
        log_eval_filepath = join(self.__log_dir, Common.log_test_eval_exp_filename)
        create_dir_if_not_exists(log_eval_filepath)
        with open(log_eval_filepath, u'w', buffering=0) as f:

            iter_messages = chain(
                [u"Results for model: {}".format(self.__model.IO.get_model_name())],
                create_experiment_eval_msgs(self.__test_results_exp_history),
                [u'--------------']
            )

            for msg in iter_messages:
                f.write(u"{}\n".format(msg))

    # endregion

    # region private methods

    def __check_costs_still_improving(self, avg_cost):
        history_len = len(self.__train_iteration_costs_history)
        if history_len <= self.__costs_window:
            return True
        return avg_cost < min(self.__train_iteration_costs_history[:history_len - self.__costs_window])

    def __is_cancel_needed_before_eval(self, avg_fit_acc):
        if self.__train_acc_limit is not None and avg_fit_acc >= self.__train_acc_limit:
            logger.info(u"Stop feeding process: avg_fit_acc > {}".format(self.__train_acc_limit))
            return True
        return False

    def __is_cancel_needed_after_eval(self, result_train, avg_fit_cost):
        """ This method related to the main algorithm that defines
            whether there is a need to stop training process or not.
        """
        assert(isinstance(result_train, TwoClassEvalResult))

        msg = None
        cancel = False

        f1_train = result_train.get_result_by_metric(result_train.C_F1)
        if self.__train_f1_limit is not None and f1_train >= self.__train_f1_limit:
            msg = u"Stop Training Process: F1-train ({f1_actual}) > {f1_bound}".format(
                f1_actual=round(f1_train, 3),
                f1_bound=self.__train_f1_limit)
            cancel = True

        if self.__key_stop_training_by_cost:
            if not self.__check_costs_still_improving(avg_fit_cost):
                msg = u"Stop Training Process: cost becomes greater than min value {} epochs ago.".format(
                    self.__costs_window)
                cancel = True

        if msg is not None:
            logger.info(msg)

        return cancel

    def __eval_and_log_results(self, operation_cancel, epoch_index, avg_fit_cost):
        assert(isinstance(operation_cancel, OperationCancellation))

        # We use the latter temporary. Maybe it might be in a way better to refactor this aspect.
        # For now such formatter could not be taken from the related experiment.
        eval_label_formatter = RuSentRelLabelsFormatter()

        result = {}

        for data_type in self.__iter_supported_data_types():
            assert(isinstance(data_type, DataType))
            result[data_type] = evaluate_model(
                experiment=self.__experiment,
                save_hidden_params=self.__key_save_hidden_parameters,
                data_type=data_type,
                epoch_index=epoch_index,
                model=self.__model,
                labels_formatter=eval_label_formatter,
                label_calc_mode=self.__label_calc_mode,
                log_dir=self.__log_dir)

        # Saving the obtained results.
        for data_type, eval_result in result.iteritems():
            self.__save_evaluation_results(result=eval_result,
                                           data_type=data_type,
                                           epoch_index=epoch_index)

        # Check whether there is a need to stop training process.
        if self.__is_cancel_needed_after_eval(result_train=result[DataType.Train], avg_fit_cost=avg_fit_cost):
            operation_cancel.Cancel()

        self.__train_iteration_costs_history.append(avg_fit_cost)
        self.__saving_results_history_optionally(result)

    def __saving_results_history_optionally(self, result):
        supported_data_types = set(self.__iter_supported_data_types())

        if DataType.Test not in supported_data_types:
            return

        iter_index = str(self.__get_iter_index())
        if iter_index not in self.__test_results_exp_history:
            self.__test_results_exp_history[iter_index] = []
        self.__test_results_exp_history[iter_index].append(result[DataType.Test])

    def __save_evaluation_results(self, result, data_type, epoch_index):
        eval_verbose_msg = create_iteration_verbose_eval_msg(eval_result=result,
                                                             data_type=data_type,
                                                             epoch_index=epoch_index)

        eval_msg = create_iteration_short_eval_msg(eval_result=result,
                                                   data_type=data_type,
                                                   epoch_index=epoch_index,
                                                   rounding_value=2)

        # Writing evaluation logging results.
        logger.info(eval_msg)

        # Separate logging information by files.
        self.__eval_iter_log_files[data_type].write(u"{}\n".format(eval_msg))
        self.__eval_iter_verbose_log_files[data_type].write(u"{}\n".format(eval_verbose_msg))

    def __get_iter_index(self):
        return self.__experiment.DocumentOperations.DataFolding.IterationIndex

    def __iter_supported_data_types(self):
        return self.__experiment.DocumentOperations.DataFolding.iter_supported_data_types()

    # endregion

    def on_fit_started(self, operation_cancel):
        if not self.__do_eval:
            return

        # Providing information into main logger.
        message = u"{}: Initial Evaluation:".format(str(datetime.datetime.now()))
        logger.info(message)

        self.__eval_and_log_results(operation_cancel=operation_cancel,
                                    epoch_index=0,
                                    avg_fit_cost=-1)

    def on_epoch_finished(self, avg_fit_cost, avg_fit_acc, epoch_index, operation_cancel):
        assert(isinstance(avg_fit_cost, float))
        assert(isinstance(avg_fit_acc, float))
        assert(isinstance(epoch_index, int))
        assert(isinstance(operation_cancel, OperationCancellation))

        message = get_message(epoch_index=epoch_index,
                              avg_fit_cost=avg_fit_cost,
                              avg_fit_acc=avg_fit_acc)

        # Providing information into main logger.
        logger.info(message)

        # Duplicate the related information in separate log file.
        self.__train_iter_log_files[DataType.Train].write(u"{}\n".format(message))

        if self.__is_cancel_needed_before_eval(avg_fit_acc):
            operation_cancel.Cancel()

        # Deciding whether there is a need in evaluation process organization.
        is_need_eval = epoch_index in self.__eval_on_epochs or operation_cancel.IsCancelled

        # Performing evaluation process (optionally).
        if self.__do_eval and is_need_eval:
            self.__eval_and_log_results(operation_cancel=operation_cancel,
                                        epoch_index=epoch_index,
                                        avg_fit_cost=avg_fit_cost)

        # Check, whether there is a need to proceed with keeping hidden states or not.
        if (epoch_index not in self.__eval_on_epochs) and (not operation_cancel.IsCancelled):
            return

        # Saving model hidden values using the related numpy utils.
        save_model_hidden_values(log_dir=self.__log_dir,
                                 epoch_index=epoch_index,
                                 save_hidden_parameters=self.__key_save_hidden_parameters,
                                 model=self.__model)

    # region base methods

    def __enter__(self):
        assert(self.__log_dir is not None)

        iter_index_int = self.__get_iter_index()
        iter_index = str(iter_index_int)

        for d_type in self.__iter_supported_data_types():

            train_log_filepath = join(self.__log_dir, Common.create_log_train_filename(iter_index=iter_index_int,
                                                                                       data_type=d_type))
            eval_log_filepath = join(self.__log_dir, Common.create_log_eval_filename(iter_index=iter_index_int,
                                                                                     data_type=d_type))
            eval_verbose_log_filepath = join(self.__log_dir, self.__log_eval_iter_verbose_filename.format(iter=iter_index,
                                                                                                          dtype=d_type))

            create_dir_if_not_exists(train_log_filepath)
            create_dir_if_not_exists(eval_log_filepath)
            create_dir_if_not_exists(eval_verbose_log_filepath)

            self.__train_iter_log_files[d_type] = open(train_log_filepath, u"w", buffering=0)
            self.__eval_iter_log_files[d_type] = open(eval_log_filepath, u"w", buffering=0)
            self.__eval_iter_verbose_log_files[d_type] = open(eval_verbose_log_filepath, u"w", buffering=0)

    def __exit__(self, exc_type, exc_val, exc_tb):

        for d_type in self.__iter_supported_data_types():

            self.__train_log_file = self.__eval_iter_log_files[d_type]
            self.__eval_log_file = self.__eval_iter_log_files[d_type]
            self.__eval_verbose_log_file = self.__eval_iter_verbose_log_files[d_type]

            if self.__train_log_file is not None:
                self.__train_log_file.close()

            if self.__eval_log_file is not None:
                self.__eval_log_file.close()

            if self.__eval_verbose_log_file is not None:
                self.__eval_verbose_log_file.close()

    # endregion