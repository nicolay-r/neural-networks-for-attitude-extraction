from arekit.contrib.networks.context.architectures.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTM
from arekit.contrib.networks.context.architectures.att_self_p_zhou_rcnn import AttentionSelfPZhouRCNN
from arekit.contrib.networks.context.architectures.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTM
from arekit.contrib.networks.context.architectures.att_self_z_yang_rcnn import AttentionSelfZYangRCNN
from arekit.contrib.networks.context.architectures.bilstm import BiLSTM
from arekit.contrib.networks.context.architectures.cnn import VanillaCNN
from arekit.contrib.networks.context.architectures.pcnn import PiecewiseCNN
from arekit.contrib.networks.context.architectures.rcnn import RCNN
from arekit.contrib.networks.context.architectures.rnn import RNN
from arekit.contrib.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arekit.contrib.networks.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig
from arekit.contrib.networks.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.contrib.networks.context.configurations.cnn import CNNConfig
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig
from arekit.contrib.networks.context.configurations.rnn import RNNConfig
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from arekit.contrib.networks.multi.architectures.base.base import BaseMultiInstanceNeuralNetwork
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from args.train.model_input_type import ModelInputType
from args.train.model_name import ModelNames


def compose_network_and_network_config_funcs(model_name, model_input_type):
    assert(isinstance(model_name, unicode))
    assert(isinstance(model_input_type, ModelInputType))

    ctx_network_func, ctx_config_func = __get_network_with_config_types(model_name)

    if model_input_type == ModelInputType.SingleInstance:
        # In case of a single instance model, there is no need to perform extra wrapping
        # since all the base models assumes to work with a single context (input).
        return ctx_network_func, ctx_config_func

    # Compose multi-instance neural network and related configuration
    # in a form of a wrapper over context-based neural network and configuration respectively.
    mi_network, mi_config = __get_mi_network_with_config(model_input_type)
    assert(issubclass(mi_network, BaseMultiInstanceNeuralNetwork))
    assert(issubclass(mi_config, BaseMultiInstanceConfig))
    return lambda: mi_network(context_network=ctx_network_func()), \
           lambda: mi_config(context_config=ctx_config_func())


# region private functions

def __get_mi_network_with_config(model_input_type):
    assert(isinstance(model_input_type, ModelInputType))

    if model_input_type == ModelInputType.MultiInstanceMaxPooling:
        return MaxPoolingOverSentences, MaxPoolingOverSentencesConfig
    if model_input_type == ModelInputType.MultiInstanceWithSelfAttention:
        return AttSelfOverSentences, AttSelfOverSentencesConfig


def __get_network_with_config_types(model_name):
    assert(isinstance(model_name, unicode))

    model_names = ModelNames()
    if model_name == model_names.SelfAttentionBiLSTM:
        return SelfAttentionBiLSTM, SelfAttentionBiLSTMConfig
    if model_name == model_names.AttSelfPZhouBiLSTM:
        return AttentionSelfPZhouBiLSTM, AttentionSelfPZhouBiLSTMConfig
    if model_name == model_names.AttSelfZYangBiLSTM:
        return AttentionSelfZYangBiLSTM, AttentionSelfZYangBiLSTMConfig
    if model_name == model_names.BiLSTM:
        return BiLSTM, BiLSTMConfig
    if model_name == model_names.CNN:
        return VanillaCNN, CNNConfig
    if model_name == model_names.LSTM:
        return RNN, RNNConfig
    if model_name == model_names.PCNN:
        return PiecewiseCNN, CNNConfig
    if model_name == model_names.RCNN:
        return RCNN, RCNNConfig
    if model_name == model_names.RCNNAttZYang:
        return AttentionSelfZYangRCNN, RCNNConfig
    if model_name == model_names.RCNNAttPZhou:
        return AttentionSelfPZhouRCNN, RCNNConfig

# endregion


def create_bags_collection_type(model_input_type):
    assert(isinstance(model_input_type, ModelInputType))

    if model_input_type == ModelInputType.SingleInstance:
        return SingleBagsCollection
    if model_input_type == ModelInputType.MultiInstanceMaxPooling:
        return MultiInstanceBagsCollection
    if model_input_type == ModelInputType.MultiInstanceWithSelfAttention:
        return MultiInstanceBagsCollection
