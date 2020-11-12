from arekit.contrib.experiments.types import ExperimentTypes
from factory_networks import INPUT_TYPE_SINGLE_INSTANCE, INPUT_TYPE_MULTI_INSTANCE
from rusentrel.classic.common import classic_ctx_common_config_settings, classic_mi_common_config_settings
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
from rusentrel.rusentrel_ds.common import ds_ctx_common_config_settings, ds_mi_common_config_settings


def get_custom_config_func(model_name, model_input_type):
    assert(isinstance(model_name, unicode))
    assert(isinstance(model_input_type, unicode))

    model_names = ModelNames()

    if model_input_type == INPUT_TYPE_SINGLE_INSTANCE:
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
            return ctx_rcnn_custom_config
        if model_name == model_names.RCNNAttZYang:
            return ctx_rcnn_z_yang_custom_config
        if model_name == model_names.RCNNAttPZhou:
            return ctx_rcnn_p_zhou_custom_config

    raise NotImplementedError()


def get_common_config_func(exp_type, model_input_type):
    assert(isinstance(exp_type, ExperimentTypes))
    assert(isinstance(model_input_type, unicode))
    if exp_type == ExperimentTypes.RuSentRel:
        if model_input_type == INPUT_TYPE_SINGLE_INSTANCE:
            return classic_ctx_common_config_settings
        if model_input_type == INPUT_TYPE_MULTI_INSTANCE:
            return classic_mi_common_config_settings
    if exp_type == ExperimentTypes.RuAttitudes:
        raise NotImplementedError
    if exp_type == ExperimentTypes.RuSentRelWithRuAttitudes:
        if model_input_type == INPUT_TYPE_SINGLE_INSTANCE:
            return ds_ctx_common_config_settings
        if model_input_type == INPUT_TYPE_MULTI_INSTANCE:
            return ds_mi_common_config_settings
