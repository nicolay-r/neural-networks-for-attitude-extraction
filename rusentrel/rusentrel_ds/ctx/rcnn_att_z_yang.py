from rusentrel.classic.ctx.rcnn_att_p_zhou import ctx_rcnn_custom_config
from rusentrel.rusentrel_ds.common import ds_ctx_common_config_settings
from arekit.contrib.networks.context.configurations.att_self_z_yang_rcnn import AttentionSelfZYangRCNNConfig
from arekit.contrib.networks.context.architectures.att_self_z_yang_rcnn import AttentionSelfZYangRCNN
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine


def run_testing_ds_rcnn_z_yang(experiment, load_model, common_callback_func):
    ExperimentEngine.run_testing(experiment=experiment,
                                 load_model=load_model,
                                 create_network=AttentionSelfZYangRCNN,
                                 create_config=AttentionSelfZYangRCNNConfig,
                                 bags_collection_type=SingleBagsCollection,
                                 common_callback_modification_func=common_callback_func,
                                 custom_config_modification_func=ctx_rcnn_custom_config,
                                 common_config_modification_func=ds_ctx_common_config_settings)
