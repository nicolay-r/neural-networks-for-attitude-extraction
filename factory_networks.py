from arekit.contrib.networks.context.architectures.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTM
from arekit.contrib.networks.context.architectures.att_self_p_zhou_rcnn import AttentionSelfPZhouRCNN
from arekit.contrib.networks.context.architectures.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTM
from arekit.contrib.networks.context.architectures.att_self_z_yang_rcnn import AttentionSelfZYangRCNN
from arekit.contrib.networks.context.architectures.bilstm import BiLSTM
from arekit.contrib.networks.context.architectures.cnn import VanillaCNN
from arekit.contrib.networks.context.architectures.pcnn import PiecewiseCNN
from arekit.contrib.networks.context.architectures.rcnn import RCNN
from arekit.contrib.networks.context.architectures.rnn import RNN
from arekit.contrib.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arekit.contrib.networks.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig
from arekit.contrib.networks.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.contrib.networks.context.configurations.cnn import CNNConfig
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig
from arekit.contrib.networks.context.configurations.rnn import RNNConfig
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from rusentrel.classic.ctx.att_self_bilstm import ctx_self_att_bilstm_custom_config
from rusentrel.classic.ctx.att_self_p_zhou import ctx_att_bilstm_p_zhou_custom_config
from rusentrel.classic.ctx.att_self_z_yang import ctx_att_bilstm_z_yang_custom_config
from rusentrel.classic.ctx.bilstm import ctx_bilstm_custom_config
from rusentrel.classic.ctx.cnn import ctx_cnn_custom_config
from rusentrel.classic.ctx.lstm import ctx_lstm_custom_config
from rusentrel.classic.ctx.pcnn import ctx_pcnn_custom_config
from rusentrel.classic.ctx.rcnn import ctx_rcnn_custom_config
from rusentrel.classic.ctx.rcnn_att_p_zhou import ctx_rcnn_p_zhou_custom_config
from rusentrel.classic.ctx.rcnn_att_z_yang import ctx_rcnn_z_yang_custom_config
from rusentrel.ctx_names import ModelNames


def get_custom_config(model_name, model_input_type):
    assert(isinstance(model_input_type, str))

    model_names = ModelNames()
    if model_name == model_names.SelfAttentionBiLSTM:
        return ctx_self_att_bilstm_custom_config
    if model_name == model_names.AttSelfPZhouBiLSTM:
        return ctx_att_bilstm_p_zhou_custom_config
    if model_name == model_names.AttSelfZYangBiLSTM:
        return ctx_att_bilstm_z_yang_custom_config
    if model_name == model_names.BiLSTM:
        return ctx_bilstm_custom_config
    if model_name == model_names.CNN:
        return ctx_cnn_custom_config
    if model_name == model_names.LSTM:
        return ctx_lstm_custom_config
    if model_name == model_names.PCNN:
        return ctx_pcnn_custom_config
    if model_name == model_names.RCNN:
        raise ctx_rcnn_custom_config
    if model_name == model_names.RCNNAttZYang:
        return ctx_rcnn_z_yang_custom_config
    if model_name == model_names.RCNNAttPZhou:
        raise ctx_rcnn_p_zhou_custom_config

    raise NotImplementedError()


def get_network_with_config(model_name):
    assert(isinstance(model_name, str))

    model_names = ModelNames()
    if model_name == model_names.SelfAttentionBiLSTM:
        return SelfAttentionBiLSTM, SelfAttentionBiLSTMConfig
    if model_name == model_names.AttSelfPZhouBiLSTM:
        return AttentionSelfPZhouBiLSTM, AttentionSelfPZhouBiLSTMConfig
    if model_name == model_names.AttSelfZYangBiLSTM:
        return AttentionSelfZYangBiLSTM, AttentionSelfZYangBiLSTMConfig
    if model_name == model_names.BiLSTM:
        return BiLSTM, BiLSTMConfig
    if model_name == model_names.CNN:
        return VanillaCNN, CNNConfig
    if model_name == model_names.LSTM:
        return RNN, RNNConfig
    if model_name == model_names.PCNN:
        return PiecewiseCNN, CNNConfig
    if model_name == model_names.RCNN:
        return RCNN, RCNNConfig
    if model_name == model_names.RCNNAttZYang:
        return AttentionSelfZYangRCNN, RCNNConfig
    if model_name == model_names.RCNNAttPZhou:
        return AttentionSelfPZhouRCNN, RCNNConfig
