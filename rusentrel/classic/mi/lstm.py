from rusentrel.classic.ctx.lstm import ctx_lstm_custom_config


# TODO. Refactor (use fixation as a common operation for a ctx-based config)
def mi_lstm_custom_config(config):
    ctx_lstm_custom_config(config.ContextConfig)
    config.fix_context_parameters()
