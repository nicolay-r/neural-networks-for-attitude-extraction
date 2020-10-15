from arekit.contrib.networks.context.architectures.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTM
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine
from rusentrel.classic.mi.att_self_z_yang import mi_att_hidden_zyang_bilstm
from rusentrel.rusentrel_ds.common import ds_common_callback_modification_func, ds_mi_common_config_settings


def run_testing_ds_mi_att_self_z_yang(experiment,
                                      load_model,
                                      network_classtype=MaxPoolingOverSentences,
                                      config_classtype=MaxPoolingOverSentencesConfig,
                                      common_callback_func=ds_common_callback_modification_func):
    ExperimentEngine.run_testing(load_model=load_model,
                                 experiment=experiment,
                                 create_network=lambda: network_classtype(context_network=AttentionSelfZYangBiLSTM()),
                                 create_config=lambda: config_classtype(context_config=AttentionSelfZYangBiLSTMConfig()),
                                 bags_collection_type=MultiInstanceBagsCollection,
                                 common_callback_modification_func=common_callback_func,
                                 custom_config_modification_func=mi_att_hidden_zyang_bilstm,
                                 common_config_modification_func=ds_mi_common_config_settings)
