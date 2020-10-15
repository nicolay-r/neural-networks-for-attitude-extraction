from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.context.architectures.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTM
from arekit.contrib.networks.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from rusentrel.rusentrel_ds.common import ds_ctx_common_config_settings
from rusentrel.classic.ctx.att_self_p_zhou import ctx_att_bilstm_custom_config


def run_testing_ds_att_bilstm_p_zhou(experiment, load_model, custom_callback_func):
    ExperimentEngine.run_testing(experiment=experiment,
                                 load_model=load_model,
                                 bags_collection_type=SingleBagsCollection,
                                 create_config=AttentionSelfPZhouBiLSTMConfig,
                                 create_network=AttentionSelfPZhouBiLSTM,
                                 common_config_modification_func=ds_ctx_common_config_settings,
                                 common_callback_modification_func=custom_callback_func,
                                 custom_config_modification_func=ctx_att_bilstm_custom_config)
