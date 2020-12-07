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

    def test(self):

        data = {
            0: [CustomEvalResult(0.1), CustomEvalResult(0.2), CustomEvalResult(0.3)],
            1: [CustomEvalResult(0.8), CustomEvalResult(0.2), CustomEvalResult(0.3), CustomEvalResult(0.3)],
            2: [CustomEvalResult(0.5), CustomEvalResult(0.2), CustomEvalResult(0.4)]
        }

        messages = create_experiment_eval_msgs(data)

        for msg in messages:
            print msg


if __name__ == '__main__':
    unittest.main()
