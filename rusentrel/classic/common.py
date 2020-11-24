from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from args.default import TEST_EVERY_K_EPOCH, EPOCHS_COUNT, BAGS_PER_MINIBATCH
from rusentrel.default import MI_CONTEXTS_PER_OPINION
from callback import NeuralNetworkCustomEvaluationCallback


def apply_classic_mi_settings(config):
    """
    Multi instance version
    """
    assert(isinstance(config, BaseMultiInstanceConfig))
    config.set_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)
    config.modify_bags_per_minibatch(BAGS_PER_MINIBATCH)


def classic_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    assert(isinstance(callback, NeuralNetworkCustomEvaluationCallback))

    callback.set_test_on_epochs(range(0, EPOCHS_COUNT + 1, TEST_EVERY_K_EPOCH))
    callback.set_key_stop_training_by_cost(False)
