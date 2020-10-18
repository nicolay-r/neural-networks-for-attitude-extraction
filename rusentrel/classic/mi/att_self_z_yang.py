from rusentrel.classic.ctx.att_self_z_yang import ctx_att_bilstm_z_yang_custom_config


# TODO. Refactor (use fixation as a common operation for a ctx-based config)
def mi_att_hidden_zyang_bilstm(config):
    ctx_att_bilstm_z_yang_custom_config(config.ContextConfig)
    config.fix_context_parameters()
