from rusentrel.classic.ctx.pcnn import ctx_pcnn_custom_config


# TODO. Refactor (use fixation as a common operation for a ctx-based config)
def mi_pcnn_custom_config(config):
    ctx_pcnn_custom_config(config.ContextConfig)
    config.fix_context_parameters()
