from arekit.contrib.networks.engine import ExperimentEngine
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.contrib.networks.context.architectures.rcnn import RCNN
from rusentrel.classic.ctx.rcnn import ctx_rcnn_custom_config
from rusentrel.classic.common import classic_mi_common_config_settings


def mi_rcnn_custom_config(config):
    ctx_rcnn_custom_config(config.ContextConfig)
    config.fix_context_parameters()


def run_mi_testing_rcnn(experiment,
                        load_model,
                        custom_callback_func,
                        network_classtype=MaxPoolingOverSentences,
                        config_classtype=MaxPoolingOverSentencesConfig,
                        custom_config_func=mi_rcnn_custom_config):

    ExperimentEngine.run_testing(
        experiment=experiment,
        load_model=load_model,
        create_network=lambda: network_classtype(context_network=RCNN()),
        create_config=lambda: config_classtype(context_config=RCNNConfig()),
        common_callback_modification_func=custom_callback_func,
        custom_config_modification_func=custom_config_func,
        common_config_modification_func=classic_mi_common_config_settings)
