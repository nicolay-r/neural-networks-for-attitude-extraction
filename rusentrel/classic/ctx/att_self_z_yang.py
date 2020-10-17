import tensorflow as tf
from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.context.architectures.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTM
from arekit.contrib.networks.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from rusentrel.classic.common import classic_ctx_common_config_settings


def ctx_att_bilstm_z_yang_custom_config(config):
    assert(isinstance(config, AttentionSelfZYangBiLSTMConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())


def run_testing_att_hidden_zyang_bilstm(experiment, load_model, custom_callback_func, create_network, create_config):
    ExperimentEngine.run_testing(
        load_model=load_model,
        create_network=create_network,
        create_config=create_config,
        experiment=experiment,
        bags_collection_type=SingleBagsCollection,
        common_callback_modification_func=custom_callback_func,
        custom_config_modification_func=ctx_att_bilstm_z_yang_custom_config,
        common_config_modification_func=classic_ctx_common_config_settings)
