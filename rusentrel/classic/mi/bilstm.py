from rusentrel.classic.ctx.bilstm import ctx_bilstm_custom_config

# TODO. Refactor (use fixation as a common operation for a ctx-based config)
def mi_bilstm_custom_config(config):
    ctx_bilstm_custom_config(config.ContextConfig)
    config.fix_context_parameters()

