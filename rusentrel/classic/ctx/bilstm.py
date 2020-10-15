from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.tf_helpers.cell_types import CellTypes
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.contrib.networks.context.architectures.bilstm import BiLSTM
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from rusentrel.classic.common import classic_ctx_common_config_settings


def ctx_bilstm_custom_config(config):
    assert(isinstance(config, BiLSTMConfig))
    config.modify_hidden_size(128)
    config.modify_bags_per_minibatch(2)
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_bags_per_minibatch(4)
    config.modify_terms_per_context(25)


def run_testing_bilstm(experiment, load_model, custom_callback_func):

    ExperimentEngine.run_testing(
        load_model=load_model,
        experiment=experiment,
        create_network=BiLSTM,
        create_config=BiLSTMConfig,
        common_callback_modification_func=custom_callback_func,
        bags_collection_type=SingleBagsCollection,
        custom_config_modification_func=ctx_bilstm_custom_config,
        common_config_modification_func=classic_ctx_common_config_settings)
