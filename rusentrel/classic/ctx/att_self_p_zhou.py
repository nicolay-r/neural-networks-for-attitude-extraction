from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.tf_helpers.cell_types import CellTypes
from arekit.contrib.networks.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig
from rusentrel.classic.common import classic_ctx_common_config_settings


def ctx_att_bilstm_p_zhou_custom_config(config):
    assert(isinstance(config, AttentionSelfPZhouBiLSTMConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_hidden_size(128)
    config.modify_cell_type(CellTypes.LSTM)
    config.modify_dropout_rnn_keep_prob(0.9)


def run_testing_att_bilstm_p_zhou(experiment, load_model, custom_callback_func,
                                  create_network, create_config):
    ExperimentEngine.run_testing(
        load_model=load_model,
        create_network=create_network,
        create_config=create_config,
        experiment=experiment,
        bags_collection_type=SingleBagsCollection,
        common_callback_modification_func=custom_callback_func,
        custom_config_modification_func=ctx_att_bilstm_p_zhou_custom_config,
        common_config_modification_func=classic_ctx_common_config_settings)
