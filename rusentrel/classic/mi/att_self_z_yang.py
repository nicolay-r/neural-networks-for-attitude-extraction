from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.context.architectures.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTM
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from rusentrel.classic.ctx.att_self_z_yang import ctx_att_hidden_zyang_bilstm_custom_config
from rusentrel.classic.common import \
    classic_common_callback_modification_func, \
    classic_mi_common_config_settings


def mi_att_hidden_zyang_bilstm(config):
    ctx_att_hidden_zyang_bilstm_custom_config(config.ContextConfig)
    config.fix_context_parameters()


def run_mi_testing_att_bilstm_z_yang(experiment,
                                     load_model,
                                     network_classtype=MaxPoolingOverSentences,
                                     config_classtype=MaxPoolingOverSentencesConfig,
                                     custom_config_func=mi_att_hidden_zyang_bilstm,
                                     custom_callback_func=classic_common_callback_modification_func):

    ExperimentEngine.run_testing(
        experiment=experiment,
        load_model=load_model,
        bags_collection_type=MultiInstanceBagsCollection,
        create_network=lambda: network_classtype(context_network=AttentionSelfZYangBiLSTM()),
        create_config=lambda: config_classtype(context_config=AttentionSelfZYangBiLSTMConfig()),
        common_callback_modification_func=custom_callback_func,
        custom_config_modification_func=custom_config_func,
        common_config_modification_func=classic_mi_common_config_settings)
