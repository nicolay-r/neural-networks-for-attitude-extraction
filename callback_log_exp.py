import collections

import numpy as np

from arekit.common.evaluation.results.base import BaseEvalResult


def create_experiment_eval_msgs(results_list_iter, iters_count):
    assert(isinstance(results_list_iter, collections.Iterable))

    f1_best_list = []
    f1_avg = None

    for results_list in results_list_iter:

        f1_per_epochs = np.array(list(__iter_f1_results(results_list)), dtype=np.float)

        # Calculate best withing every iteration.
        f1_best_list.append(max(f1_per_epochs))

        # Calculate average
        if f1_avg is None:
            f1_avg = np.zeros(f1_per_epochs.size, dtype=np.float)
        f1_avg += f1_per_epochs

    f1_avg /= iters_count

    f1_best_avg = float(sum(f1_best_list)) / len(f1_best_list)

    messages_list = [
        u"F1-last avg.: {}".format(round(f1_avg[-1], 2)),
        u"F1 per epoch avg.: {}".format([round(f1, 2) for f1 in f1_avg]),
        u"F1-best avg.: {}".format(round(f1_best_avg, 3)),
        u"F1-best per iterations: {}".format([round(f1_best, 3) for f1_best in f1_best_list])
    ]

    return messages_list


def __iter_f1_results(results_list):
    assert(isinstance(results_list, list))
    for result in results_list:
        assert(isinstance(result, BaseEvalResult))
        yield result.get_result_by_metric(result.C_F1)
