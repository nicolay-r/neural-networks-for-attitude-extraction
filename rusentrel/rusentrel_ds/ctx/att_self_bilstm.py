from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from rusentrel.rusentrel_ds.common import ds_ctx_common_config_settings, ds_common_callback_modification_func
from rusentrel.classic.ctx.att_self_bilstm import ctx_self_att_bilstm_custom_config


def run_testing_ds_self_att_bilstm(experiment, common_callback_func=ds_common_callback_modification_func):
    ExperimentEngine.run_testing(experiment=experiment,
                                 create_network=SelfAttentionBiLSTM,
                                 create_config=SelfAttentionBiLSTMConfig,
                                 bags_collection_type=SingleBagsCollection,
                                 common_config_modification_func=ds_ctx_common_config_settings,
                                 common_callback_modification_func=common_callback_func,
                                 custom_config_modification_func=ctx_self_att_bilstm_custom_config)
