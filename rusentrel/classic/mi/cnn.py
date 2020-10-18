from rusentrel.classic.ctx.cnn import ctx_cnn_custom_config

# TODO. Refactor (use fixation as a common operation for a ctx-based config)
def mi_cnn_custom_config(config):
    ctx_cnn_custom_config(config.ContextConfig)
    config.fix_context_parameters()
