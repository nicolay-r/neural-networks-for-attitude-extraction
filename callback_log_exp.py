import numpy as np

from arekit.common.evaluation.results.base import BaseEvalResult


def create_experiment_eval_msgs(results_per_epoch):
    assert(isinstance(results_per_epoch, dict))

    f1_best_list = []

    max_epochs_count = max([len(results_list) for results_list in results_per_epoch.itervalues()])
    f1_avg = np.zeros(max_epochs_count, dtype=np.float)

    for results_list in results_per_epoch.itervalues():

        # Compose and proceed the further epochs with the last known result.
        f1_per_epochs_list = list(__iter_f1_results(results_list))
        while len(f1_per_epochs_list) < max_epochs_count:
            f1_per_epochs_list.append(f1_per_epochs_list[-1])

        f1_per_epochs = np.array(f1_per_epochs_list, dtype=np.float)

        # Calculate best withing every iteration.
        f1_best_list.append(max(f1_per_epochs))

        # Calculate average
        f1_avg += f1_per_epochs

    f1_avg /= len(results_per_epoch)

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
