from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.context.architectures.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTM
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from rusentrel.classic.ctx.att_self_p_zhou import ctx_att_bilstm_custom_config
from rusentrel.classic.common import classic_mi_common_config_settings, classic_common_callback_modification_func


def mi_att_bilstm_custom_config(config):
    ctx_att_bilstm_custom_config(config.ContextConfig)
    config.fix_context_parameters()


def run_mi_testing_att_bilstm_p_zhou(experiment,
                                     load_model,
                                     network_classtype=MaxPoolingOverSentences,
                                     config_classtype=MaxPoolingOverSentencesConfig,
                                     custom_config_func=mi_att_bilstm_custom_config,
                                     custom_callback_func=classic_common_callback_modification_func):

    ExperimentEngine.run_testing(
        experiment=experiment,
        load_model=load_model,
        create_network=lambda: network_classtype(context_network=AttentionSelfPZhouBiLSTM()),
        create_config=lambda: config_classtype(context_config=AttentionSelfPZhouBiLSTMConfig()),
        bags_collection_type=MultiInstanceBagsCollection,
        common_callback_modification_func=custom_callback_func,
        custom_config_modification_func=custom_config_func,
        common_config_modification_func=classic_mi_common_config_settings)
