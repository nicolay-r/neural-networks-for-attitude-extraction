from arekit.contrib.networks.context.configurations.rnn import RNNConfig
from arekit.contrib.networks.context.architectures.rnn import RNN
from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.tf_helpers.cell_types import CellTypes
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from rusentrel.classic.common import classic_ctx_common_config_settings


def ctx_lstm_custom_config(config):
    assert(isinstance(config, RNNConfig))
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_hidden_size(128)
    config.modify_bags_per_minibatch(2)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_learning_rate(0.1)
    config.modify_terms_per_context(25)


def run_testing_lstm(experiment, load_model, custom_callback_func):
    ExperimentEngine.run_testing(
        load_model=load_model,
        experiment=experiment,
        create_network=RNN,
        create_config=RNNConfig,
        bags_collection_type=SingleBagsCollection,
        common_callback_modification_func=custom_callback_func,
        custom_config_modification_func=ctx_lstm_custom_config,
        common_config_modification_func=classic_ctx_common_config_settings)
