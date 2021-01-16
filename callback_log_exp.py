from collections import OrderedDict

import numpy as np

from arekit.common.evaluation.results.base import BaseEvalResult


OUTPUT_PRECISION = 3


def create_experiment_eval_msgs(results_per_epoch):
    assert(isinstance(results_per_epoch, OrderedDict))

    if len(results_per_epoch) == 0:
        return [u"No Iterations"]

    f1_last = []
    f1_best_list = []

    max_epochs_count = max([len(results_list) for results_list in results_per_epoch.itervalues()])
    f1_avg = np.zeros(max_epochs_count, dtype=np.float)

    if max_epochs_count == 0:
        return [u"No Results"]

    for results_list in results_per_epoch.itervalues():

        # Compose and proceed the further epochs with the last known result.
        f1_per_epochs_list = list(__iter_f1_results(results_list))
        while len(f1_per_epochs_list) < max_epochs_count:
            f1_per_epochs_list.append(f1_per_epochs_list[-1])

        f1_per_epochs = np.array(f1_per_epochs_list, dtype=np.float)

        # Calculate best within every iteration.
        f1_best_list.append(max(f1_per_epochs))

        # Calculate average
        f1_avg += f1_per_epochs

        # Take last
        f1_last.append(f1_per_epochs_list[-1])

    f1_avg /= len(results_per_epoch)

    f1_best_avg = float(sum(f1_best_list)) / len(f1_best_list)

    return [
        u"F1-last avg.: {}".format(round(np.mean(f1_last), OUTPUT_PRECISION)),
        u"F1-last per iterations: {}".format([round(f1, OUTPUT_PRECISION) for f1 in f1_last]),
        u"F1-avg. per epoch: {}".format([round(f1, OUTPUT_PRECISION) for f1 in f1_avg]),
        u"F1-best avg.: {}".format(round(f1_best_avg, OUTPUT_PRECISION)),
        u"F1-best per iterations: {}".format([round(f1_best, OUTPUT_PRECISION) for f1_best in f1_best_list])
    ]


def parse_last_epoch_results(filepath):
    """ Perform results reading from the related filepath
    """
    # Example to parse:
    # F1-last per iterations: [0.32, 0.229, 0.311]

    iters = None
    with open(filepath) as f:
        for line in f.readlines():
            if u'last per iterations' in line:
                arr = line.split(':')[1].strip()
                vals = arr[1:-1]
                iters = [float(v) for v in vals.split(',')]

    return iters


def __iter_f1_results(results_list):
    assert(isinstance(results_list, list))
    for result in results_list:
        assert(isinstance(result, BaseEvalResult))
        yield result.get_result_by_metric(result.C_F1)
