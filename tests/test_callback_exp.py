import unittest
from collections import OrderedDict

from arekit.common.evaluation.results.base import BaseEvalResult
from callback_log_exp import create_experiment_eval_msgs
from callback_log_training import get_message


class CustomEvalResult(BaseEvalResult):

    def __init__(self, x):
        super(CustomEvalResult, self).__init__()
        self.__x = x

    def get_result_by_metric(self, metric_name):
        return self.__x


class TestCallbackExperimentEvaluationOutput(unittest.TestCase):

    def test_empty_data(self):
        for msg in create_experiment_eval_msgs(OrderedDict()):
            print msg

    def test_empty_iteration_results(self):
        for msg in create_experiment_eval_msgs(OrderedDict({0: [], 1: []})):
            print msg

    def test_get_msg(self):
        print get_message(epoch_index=0, avg_fit_acc=1, avg_fit_cost=0.5)

    def test_var_length(self):

        data = OrderedDict({
            0: [CustomEvalResult(0.112), CustomEvalResult(0.212), CustomEvalResult(0.333)],
            1: [CustomEvalResult(0.811), CustomEvalResult(0.210), CustomEvalResult(0.333), CustomEvalResult(0.3)],
            2: [CustomEvalResult(0.511), CustomEvalResult(0.210), CustomEvalResult(0.433)]
        })

        for msg in create_experiment_eval_msgs(data):
            print msg


if __name__ == '__main__':
    unittest.main()
