from rusentrel.classic.ctx.att_self_p_zhou import ctx_att_bilstm_p_zhou_custom_config


# TODO. Refactor (use fixation as a common operation for a ctx-based config)
def mi_att_bilstm_custom_config(config):
    ctx_att_bilstm_p_zhou_custom_config(config.ContextConfig)
    config.fix_context_parameters()
