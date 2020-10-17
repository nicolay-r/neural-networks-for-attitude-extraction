import tensorflow as tf
from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.context.configurations.cnn import CNNConfig
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from rusentrel.classic.common import classic_ctx_common_config_settings


def ctx_cnn_custom_config(config):
    assert(isinstance(config, CNNConfig))
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())


def run_testing_cnn(experiment, load_model, custom_callback_func, create_network, create_config, custom_config):
    ExperimentEngine.run_testing(
        load_model=load_model,
        create_network=create_network,
        create_config=create_config,
        experiment=experiment,
        bags_collection_type=SingleBagsCollection,
        common_callback_modification_func=custom_callback_func,
        custom_config_modification_func=custom_config,
        common_config_modification_func=classic_ctx_common_config_settings)

