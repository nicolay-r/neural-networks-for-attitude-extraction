from arekit.contrib.networks.context.configurations.rnn import RNNConfig
from arekit.contrib.networks.context.architectures.rnn import RNN
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine
from rusentrel.rusentrel_ds.common import ds_ctx_common_config_settings, ds_common_callback_modification_func
from rusentrel.classic.ctx.lstm import ctx_lstm_custom_config


def run_testing_ds_lstm(experiment, common_callback_func=ds_common_callback_modification_func):
    ExperimentEngine.run_testing(create_network=RNN,
                                 create_config=RNNConfig,
                                 experiment=experiment,
                                 bags_collection_type=SingleBagsCollection,
                                 common_config_modification_func=ds_ctx_common_config_settings,
                                 common_callback_modification_func=common_callback_func,
                                 custom_config_modification_func=ctx_lstm_custom_config)
