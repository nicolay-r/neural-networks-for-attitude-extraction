from arekit.contrib.networks.context.architectures.rcnn import RCNN
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine
from rusentrel.rusentrel_ds.common import ds_ctx_common_config_settings
from rusentrel.classic.ctx.rcnn import ctx_rcnn_custom_config


def run_testing_ds_rcnn(experiment, load_model, common_callback_func):
    ExperimentEngine.run_testing(experiment=experiment,
                                 create_network=RCNN,
                                 create_config=RCNNConfig,
                                 load_model=load_model,
                                 bags_collection_type=SingleBagsCollection,
                                 common_config_modification_func=ds_ctx_common_config_settings,
                                 common_callback_modification_func=common_callback_func,
                                 custom_config_modification_func=ctx_rcnn_custom_config)
