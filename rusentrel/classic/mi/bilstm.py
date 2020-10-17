from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.contrib.networks.context.architectures.bilstm import BiLSTM
from rusentrel.classic.ctx.bilstm import ctx_bilstm_custom_config
from rusentrel.classic.common import classic_mi_common_config_settings


def mi_bilstm_custom_config(config):
    ctx_bilstm_custom_config(config.ContextConfig)
    config.fix_context_parameters()


def run_mi_testing_bilstm(experiment,
                          load_model,
                          custom_callback_func,
                          network_classtype=MaxPoolingOverSentences,
                          config_classtype=MaxPoolingOverSentencesConfig,
                          custom_config_func=mi_bilstm_custom_config):

    ExperimentEngine.run_testing(
        load_model=load_model,
        create_network=lambda: network_classtype(context_network=BiLSTM()),
        create_config=lambda: config_classtype(context_config=BiLSTMConfig()),
        experiment=experiment,
        bags_collection_type=MultiInstanceBagsCollection,
        common_callback_modification_func=custom_callback_func,
        custom_config_modification_func=custom_config_func,
        common_config_modification_func=classic_mi_common_config_settings)

