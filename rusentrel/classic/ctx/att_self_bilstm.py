from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.tf_helpers.cell_types import CellTypes
from arekit.contrib.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from rusentrel.classic.common import classic_ctx_common_config_settings


def ctx_self_att_bilstm_custom_config(config):
    assert(isinstance(config, SelfAttentionBiLSTMConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_penaltization_term_coef(0.5)
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_terms_per_context(25)


def run_testing_self_att_bilstm(experiment, load_model, custom_callback_func):
    ExperimentEngine.run_testing(
        load_model=load_model,
        experiment=experiment,
        create_network=SelfAttentionBiLSTM,
        create_config=SelfAttentionBiLSTMConfig,
        bags_collection_type=SingleBagsCollection,
        common_callback_modification_func=custom_callback_func,
        custom_config_modification_func=ctx_self_att_bilstm_custom_config,
        common_config_modification_func=classic_ctx_common_config_settings)
