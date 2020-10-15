from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.contrib.networks.context.architectures.bilstm import BiLSTM
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine
from rusentrel.rusentrel_ds.common import ds_ctx_common_config_settings, ds_common_callback_modification_func
from rusentrel.classic.ctx.bilstm import ctx_bilstm_custom_config


def run_testing_ds_bilstm(experiment, load_model, common_callback_func=ds_common_callback_modification_func):
    ExperimentEngine.run_testing(experiment=experiment,
                                 load_model=load_model,
                                 create_network=BiLSTM,
                                 create_config=BiLSTMConfig,
                                 bags_collection_type=SingleBagsCollection,
                                 common_config_modification_func=ds_ctx_common_config_settings,
                                 common_callback_modification_func=common_callback_func,
                                 custom_config_modification_func=ctx_bilstm_custom_config)
