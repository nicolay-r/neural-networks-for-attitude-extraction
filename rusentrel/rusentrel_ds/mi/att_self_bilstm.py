from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from rusentrel.classic.mi.att_self_bilstm import mi_self_att_bilstm_custom_config
from rusentrel.rusentrel_ds.common import ds_common_callback_modification_func, ds_mi_common_config_settings


def run_testing_ds_mi_self_att_bilstm(experiment,
                                      load_model,
                                      network_classtype=MaxPoolingOverSentences,
                                      config_classtype=MaxPoolingOverSentencesConfig,
                                      common_callback_func=ds_common_callback_modification_func):
    ExperimentEngine.run_testing(experiment=experiment,
                                 load_model=load_model,
                                 create_network=lambda: network_classtype(context_network=SelfAttentionBiLSTM()),
                                 create_config=lambda: config_classtype(context_config=SelfAttentionBiLSTMConfig()),
                                 bags_collection_type=MultiInstanceBagsCollection,
                                 common_callback_modification_func=common_callback_func,
                                 custom_config_modification_func=mi_self_att_bilstm_custom_config,
                                 common_config_modification_func=ds_mi_common_config_settings)
