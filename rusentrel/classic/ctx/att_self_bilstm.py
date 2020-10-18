from arekit.contrib.networks.tf_helpers.cell_types import CellTypes
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig


def ctx_self_att_bilstm_custom_config(config):
    assert(isinstance(config, SelfAttentionBiLSTMConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_penaltization_term_coef(0.5)
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_terms_per_context(25)
