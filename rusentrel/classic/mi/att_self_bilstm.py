from rusentrel.classic.ctx.att_self_bilstm import ctx_self_att_bilstm_custom_config

# TODO. Refactor (use fixation as a common operation for a ctx-based config)
def mi_self_att_bilstm_custom_config(config):
    ctx_self_att_bilstm_custom_config(config.ContextConfig)
    config.fix_context_parameters()
