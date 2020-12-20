from arekit.contrib.experiments.types import ExperimentTypes
from args.train.model_input_type import ModelInputType
from rusentrel.classic.common import apply_classic_mi_settings
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
from args.train.model_name import ModelNames
from rusentrel.rusentrel_ds.common import apply_ds_mi_settings


def modify_config_for_model(model_name, model_input_type, config):
    assert(isinstance(model_name, unicode))
    assert(isinstance(model_input_type, ModelInputType))

    model_names = ModelNames()

    if model_input_type == ModelInputType.SingleInstance:
        if model_name == model_names.SelfAttentionBiLSTM:
            ctx_self_att_bilstm_custom_config(config)
        if model_name == model_names.AttSelfPZhouBiLSTM:
            ctx_att_bilstm_p_zhou_custom_config(config)
        if model_name == model_names.AttSelfZYangBiLSTM:
            ctx_att_bilstm_z_yang_custom_config(config)
        if model_name == model_names.BiLSTM:
            ctx_bilstm_custom_config(config)
        if model_name == model_names.CNN:
            ctx_cnn_custom_config(config)
        if model_name == model_names.LSTM:
            ctx_lstm_custom_config(config)
        if model_name == model_names.PCNN:
            ctx_pcnn_custom_config(config)
        if model_name == model_names.RCNN:
            ctx_rcnn_custom_config(config)
        if model_name == model_names.RCNNAttZYang:
            ctx_rcnn_z_yang_custom_config(config)
        if model_name == model_names.RCNNAttPZhou:
            ctx_rcnn_p_zhou_custom_config(config)

        return

    if model_input_type == ModelInputType.MultiInstance:
        # We assign all the settings related to the case of
        # single instance model.
        modify_config_for_model(model_name=model_name,
                                model_input_type=ModelInputType.SingleInstance,
                                config=config)
        # We apply modification of some parameters
        config.fix_context_parameters()

        return

    raise NotImplementedError(u"Model input type {input_type} is not supported".format(
        input_type=model_input_type))


def optionally_modify_config_for_experiment(config, exp_type, model_input_type):
    assert(isinstance(exp_type, ExperimentTypes))
    assert(isinstance(model_input_type, ModelInputType))

    if model_input_type == ModelInputType.MultiInstance:
        if exp_type == ExperimentTypes.RuSentRel:
            apply_classic_mi_settings(config)
        if exp_type == ExperimentTypes.RuAttitudes or exp_type == ExperimentTypes.RuSentRelWithRuAttitudes:
            apply_ds_mi_settings(config)

        return
