from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from callback import CustomCallback
from rusentrel.classic.common import classic_ctx_common_config_settings
from rusentrel.default import MI_CONTEXTS_PER_OPINION

DS_NAME_PREFIX = u'ds_'


def ds_ctx_common_config_settings(config):
    """
    This function describes a base config setup for all models.
    """
    assert(isinstance(config, DefaultNetworkConfig))

    # Apply classic settings
    classic_ctx_common_config_settings(config)


def ds_mi_common_config_settings(config):
    """
    This function describes a base config setup for all models.
    """
    assert(isinstance(config, BaseMultiInstanceConfig))
    ds_ctx_common_config_settings(config)

    # Increasing memory limit consumption
    config.set_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)
    config.modify_bags_per_minibatch(2)


def ds_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    assert(isinstance(callback, CustomCallback))

    callback.set_test_on_epochs(range(0, 50, 5))
    callback.set_cancellation_acc_bound(0.999)
    callback.set_cancellation_f1_train_bound(0.86)
    callback.set_key_stop_training_by_cost(False)
