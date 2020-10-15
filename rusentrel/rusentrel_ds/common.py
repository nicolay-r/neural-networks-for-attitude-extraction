from arekit.contrib.experiments.rusentrel_ds.experiment import RuSentRelWithRuAttitudesExperiment
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions

from callback import CustomCallback
from io_utils import RuSentRelBasedExperimentsIOUtils
from rusentrel.base import data_io_post_initialization

from rusentrel.classic.common import classic_ctx_common_config_settings
from rusentrel.default import MI_CONTEXTS_PER_OPINION

DS_NAME_PREFIX = u'ds_'

# region ds config settings


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


# endregion

# region ds callback settings

def ds_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    assert(isinstance(callback, CustomCallback))

    callback.set_test_on_epochs(range(0, 50, 5))
    callback.set_cancellation_acc_bound(0.999)
    callback.set_cancellation_f1_train_bound(0.86)
    callback.set_key_stop_training_by_cost(False)

# endregion


def create_rsr_ds_experiment(data_io, version=RuAttitudesVersions.V20):
    assert(isinstance(data_io, RuSentRelBasedExperimentsIOUtils))

    return RuSentRelWithRuAttitudesExperiment(version=version,
                                              data_io=data_io,
                                              rusentrel_version=data_io.RuSentRelVersion,
                                              prepare_model_root=True)
