import logging

from arekit.common.experiment.data_type import DataType
from arekit.contrib.networks.core.callback.network import NeuralNetworkCallback
from arekit.contrib.networks.core.cancellation import OperationCancellation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CustomCallback(NeuralNetworkCallback):

    def __init__(self):
        super(CustomCallback, self).__init__()

        self.__costs_history = None
        self.__costs_window = 5
        self.__cancellation_acc_bound = 0.99
        self.__cancellation_f1_train_bound = 0.85

    # region public methods

    def reset_experiment_dependent_parameters(self):
        self.__costs_history = []

    def set_cancellation_acc_bound(self, value):
        assert(isinstance(value, float))
        self.__cancellation_acc_bound = value

    def set_cancellation_f1_train_bound(self, value):
        assert(isinstance(value, float))
        self.__cancellation_f1_train_bound = value

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

    # endregion

    def on_epoch_finished(self, avg_fit_cost, avg_fit_acc, epoch_index, operation_cancel):
        assert(isinstance(avg_fit_cost, float))
        assert(isinstance(avg_fit_acc, float))
        assert(isinstance(epoch_index, int))
        assert(isinstance(operation_cancel, OperationCancellation))

        if avg_fit_acc >= self.__cancellation_acc_bound:
            logger.info("Stop feeding process: avg_fit_acc > {}".format(self.__cancellation_acc_bound))
            operation_cancel.Cancel()

        if epoch_index in self._test_on_epochs or operation_cancel.IsCancelled:
            self.__test(operation_cancel=operation_cancel,
                        epoch_index=epoch_index,
                        avg_fit_cost=avg_fit_cost)

        # Running base method
        super(CustomCallback, self).on_epoch_finished(
            avg_fit_cost=avg_fit_cost,
            avg_fit_acc=avg_fit_acc,
            epoch_index=epoch_index,
            operation_cancel=operation_cancel)

    def __test(self, operation_cancel, epoch_index, avg_fit_cost):
        result_train = self._evaluate_model(data_type=DataType.Train,
                                            epoch_index=epoch_index)

        f1_train = result_train.get_result_by_metric(result_train.C_F1)
        if f1_train >= self.__cancellation_f1_train_bound:

            logger.info("Stop feeding process: F1-train ({}) > {}".format(
                round(f1_train, 3),
                self.__cancellation_f1_train_bound))

            operation_cancel.Cancel()

        self._evaluate_model(data_type=DataType.Test,
                             epoch_index=epoch_index)

        if self.__key_stop_training_by_cost:
            if not self.__check_costs_still_improving(avg_fit_cost):
                print "Cancelling: cost becomes greater than min value {} epochs ago.".format(
                    self.__costs_window)
                operation_cancel.Cancel()

        self.__costs_history.append(avg_fit_cost)
