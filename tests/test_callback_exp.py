import unittest

from arekit.common.evaluation.results.base import BaseEvalResult
from callback_log_exp import create_experiment_eval_msgs


class CustomEvalResult(BaseEvalResult):

    def __init__(self, x):
        super(CustomEvalResult, self).__init__()
        self.__x = x

    def get_result_by_metric(self, metric_name):
        return self.__x


class TestCallbackExperimentEvaluationOutput(unittest.TestCase):

    def test_empty_data(self):
        for msg in create_experiment_eval_msgs({}):
            print msg

    def test_empty_iteration_results(self):
        for msg in create_experiment_eval_msgs({0: [], 1: []}):
            print msg

    def test_var_length(self):

        data = {
            0: [CustomEvalResult(0.112), CustomEvalResult(0.212), CustomEvalResult(0.333)],
            1: [CustomEvalResult(0.811), CustomEvalResult(0.210), CustomEvalResult(0.333), CustomEvalResult(0.3)],
            2: [CustomEvalResult(0.511), CustomEvalResult(0.210), CustomEvalResult(0.433)]
        }

        for msg in create_experiment_eval_msgs(data):
            print msg


if __name__ == '__main__':
    unittest.main()
