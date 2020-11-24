from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from callback import NeuralNetworkCustomEvaluationCallback
from rusentrel.default import MI_CONTEXTS_PER_OPINION


def apply_ds_mi_settings(config):
    """
    This function describes a base config setup for all models.
    """
    assert(isinstance(config, BaseMultiInstanceConfig))
    config.set_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)
    config.modify_bags_per_minibatch(2)


def ds_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    assert(isinstance(callback, NeuralNetworkCustomEvaluationCallback))

    callback.set_test_on_epochs(range(0, 50, 5))
    callback.set_cancellation_acc_bound(0.999)
    callback.set_cancellation_f1_train_bound(0.86)
    callback.set_key_stop_training_by_cost(False)
