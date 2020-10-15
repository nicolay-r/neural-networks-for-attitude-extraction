from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from rusentrel.default import MI_CONTEXTS_PER_OPINION
from callback import CustomCallback
from io_utils import RuSentRelBasedExperimentsIOUtils


def classic_ctx_common_config_settings(config):
    """
    Context version
    """
    assert(isinstance(config, DefaultNetworkConfig))

    config.modify_learning_rate(0.1)
    config.modify_use_class_weights(True)
    config.modify_dropout_keep_prob(0.5)
    config.modify_bag_size(1)
    config.modify_bags_per_minibatch(8)
    config.modify_embedding_dropout_keep_prob(1.0)
    config.modify_terms_per_context(50)
    config.modify_use_entity_types_in_embedding(False)


def classic_mi_common_config_settings(config):
    """
    Multi instance version
    """
    assert(isinstance(config, BaseMultiInstanceConfig))
    classic_ctx_common_config_settings(config)
    config.set_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)
    config.modify_bags_per_minibatch(2)


def classic_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    assert(isinstance(callback, CustomCallback))

    callback.set_test_on_epochs(range(0, 151, 10))
    callback.set_key_stop_training_by_cost(False)


def create_rusentrel_experiment(data_io):
    assert(isinstance(data_io, RuSentRelBasedExperimentsIOUtils))
    return RuSentRelExperiment(data_io=data_io,
                               version=data_io.RuSentRelVersion,
                               prepare_model_root=True)
