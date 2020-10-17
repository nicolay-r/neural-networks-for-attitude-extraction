from arekit.contrib.networks.context.architectures.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTM
from arekit.contrib.networks.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine
from rusentrel.rusentrel_ds.common import ds_ctx_common_config_settings
from rusentrel.classic.ctx.att_self_z_yang import ctx_att_bilstm_z_yang_custom_config


def run_testing_ds_att_hidden_zyang_bilstm(experiment, load_model, custom_callback_func):
    ExperimentEngine.run_testing(experiment=experiment,
                                 load_model=load_model,
                                 create_network=AttentionSelfZYangBiLSTM,
                                 create_config=AttentionSelfZYangBiLSTMConfig,
                                 bags_collection_type=SingleBagsCollection,
                                 common_config_modification_func=ds_ctx_common_config_settings,
                                 common_callback_modification_func=custom_callback_func,
                                 custom_config_modification_func=ctx_att_bilstm_z_yang_custom_config)
