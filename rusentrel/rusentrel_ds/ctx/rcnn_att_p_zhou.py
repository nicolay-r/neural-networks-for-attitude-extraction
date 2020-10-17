from rusentrel.classic.ctx.rcnn_att_p_zhou import ctx_rcnn_p_zhou_custom_config
from rusentrel.rusentrel_ds.common import ds_ctx_common_config_settings
from arekit.contrib.networks.context.architectures.att_self_p_zhou_rcnn import AttentionSelfPZhouRCNN
from arekit.contrib.networks.context.configurations.att_self_p_zhou_rcnn import AttentionSelfPZhouRCNNConfig
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine


def run_testing_ds_rcnn_p_zhou(experiment, load_model, common_callback_func):
    ExperimentEngine.run_testing(experiment=experiment,
                                 load_model=load_model,
                                 create_network=AttentionSelfPZhouRCNN,
                                 create_config=AttentionSelfPZhouRCNNConfig,
                                 bags_collection_type=SingleBagsCollection,
                                 common_callback_modification_func=common_callback_func,
                                 custom_config_modification_func=ctx_rcnn_p_zhou_custom_config,
                                 common_config_modification_func=ds_ctx_common_config_settings)
