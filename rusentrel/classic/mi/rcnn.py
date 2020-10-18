from rusentrel.classic.ctx.rcnn import ctx_rcnn_custom_config


# TODO. Refactor (use fixation as a common operation for a ctx-based config)
def mi_rcnn_custom_config(config):
    ctx_rcnn_custom_config(config.ContextConfig)
    config.fix_context_parameters()
