from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from args.default import TERMS_PER_CONTEXT, DROPOUT_KEEP_PROB, TEST_EVERY_K_EPOCH, EPOCHS_COUNT, BAGS_PER_MINIBATCH, \
    BAG_SIZE
from rusentrel.default import MI_CONTEXTS_PER_OPINION
from callback import CustomCallback


def classic_ctx_common_config_settings(config):
    """
    Context version
    """
    assert(isinstance(config, DefaultNetworkConfig))

    # TODO. Provide the related parameters from cmd.

    config.modify_learning_rate(0.1)
    config.modify_use_class_weights(True)
    config.modify_dropout_keep_prob(DROPOUT_KEEP_PROB)
    config.modify_bag_size(BAG_SIZE)
    config.modify_bags_per_minibatch(BAGS_PER_MINIBATCH)
    config.modify_embedding_dropout_keep_prob(1.0)
    config.modify_terms_per_context(TERMS_PER_CONTEXT)
    config.modify_use_entity_types_in_embedding(False)


def classic_mi_common_config_settings(config):
    """
    Multi instance version
    """
    assert(isinstance(config, BaseMultiInstanceConfig))
    classic_ctx_common_config_settings(config)
    config.set_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)
    config.modify_bags_per_minibatch(BAGS_PER_MINIBATCH)


def classic_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    assert(isinstance(callback, CustomCallback))

    callback.set_test_on_epochs(range(0, EPOCHS_COUNT + 1, TEST_EVERY_K_EPOCH))
    callback.set_key_stop_training_by_cost(False)
