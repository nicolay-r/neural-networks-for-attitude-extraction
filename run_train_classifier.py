import argparse
from args.cv_index import CvCountArg
from args.experiment import ExperimentTypeArg, SUPERVISED_LEARNING, SUPERVISED_LEARNING_WITH_DS
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg
from common import Common
from io_utils import RuSentRelBasedExperimentsIOUtils
from rusentrel.classic.common import classic_common_callback_modification_func
from rusentrel.classic.ctx.att_self_bilstm import run_testing_self_att_bilstm
from rusentrel.classic.ctx.att_self_p_zhou import run_testing_att_bilstm_p_zhou
from rusentrel.classic.ctx.att_self_z_yang import run_testing_att_hidden_zyang_bilstm
from rusentrel.classic.ctx.bilstm import run_testing_bilstm
from rusentrel.classic.ctx.cnn import run_testing_cnn
from rusentrel.classic.ctx.lstm import run_testing_lstm
from rusentrel.classic.ctx.pcnn import run_testing_pcnn
from rusentrel.classic.ctx.rcnn import run_testing_rcnn
from rusentrel.classic.ctx.rcnn_att_p_zhou import run_testing_rcnn_p_zhou
from rusentrel.classic.ctx.rcnn_att_z_yang import run_testing_rcnn_z_yang
from rusentrel.classic.mi.att_self_bilstm import run_mi_testing_self_att_bilstm
from rusentrel.classic.mi.att_self_p_zhou import run_mi_testing_att_bilstm_p_zhou
from rusentrel.classic.mi.att_self_z_yang import run_mi_testing_att_bilstm_z_yang
from rusentrel.classic.mi.cnn import run_mi_testing_cnn
from rusentrel.classic.mi.lstm import run_mi_testing_lstm
from rusentrel.classic.mi.pcnn import run_mi_testing_pcnn
from rusentrel.classic.mi.rcnn import run_mi_testing_rcnn
from rusentrel.callback_utils import classic_cv_common_callback_modification_func, \
    ds_cv_common_callback_modification_func
from rusentrel.ctx_names import ModelNames
from rusentrel.mi_names import MaxPoolingModelNames
from rusentrel.rusentrel_ds.common import ds_common_callback_modification_func
from rusentrel.rusentrel_ds.ctx.att_self_bilstm import run_testing_ds_self_att_bilstm
from rusentrel.rusentrel_ds.ctx.att_self_p_zhou import run_testing_ds_att_bilstm_p_zhou
from rusentrel.rusentrel_ds.ctx.att_self_z_yang import run_testing_ds_att_hidden_zyang_bilstm
from rusentrel.rusentrel_ds.ctx.bilstm import run_testing_ds_bilstm
from rusentrel.rusentrel_ds.ctx.cnn import run_testing_ds_cnn
from rusentrel.rusentrel_ds.ctx.lstm import run_testing_ds_lstm
from rusentrel.rusentrel_ds.ctx.pcnn import run_testing_ds_pcnn
from rusentrel.rusentrel_ds.ctx.rcnn import run_testing_ds_rcnn
from rusentrel.rusentrel_ds.ctx.rcnn_att_p_zhou import run_testing_ds_rcnn_p_zhou
from rusentrel.rusentrel_ds.ctx.rcnn_att_z_yang import run_testing_ds_rcnn_z_yang
from rusentrel.rusentrel_ds.mi.att_self_bilstm import run_testing_ds_mi_self_att_bilstm
from rusentrel.rusentrel_ds.mi.att_self_p_zhou import run_testing_ds_mi_att_self_p_zhou
from rusentrel.rusentrel_ds.mi.att_self_z_yang import run_testing_ds_mi_att_self_z_yang
from rusentrel.rusentrel_ds.mi.cnn import run_testing_ds_mi_cnn
from rusentrel.rusentrel_ds.mi.lstm import run_testing_ds_mi_lstm
from rusentrel.rusentrel_ds.mi.pcnn import run_testing_ds_mi_pcnn
from rusentrel.rusentrel_ds.mi.rcnn import run_testing_ds_mi_rcnn

INPUT_TYPE_SINGLE_INSTANCE = 'ctx'
INPUT_TYPE_MULTI_INSTANCE = 'mi'
INPUT_TYPE_MULTI_INSTANCE_WITH_ATTENTION = 'mi'


def supported_model_names():
    model_names = ModelNames()
    return [
        model_names.SelfAttentionBiLSTM,
        model_names.AttSelfPZhouBiLSTM,
        model_names.AttSelfZYangBiLSTM,
        model_names.BiLSTM,
        model_names.CNN,
        model_names.LSTM,
        model_names.PCNN,
        model_names.RCNN,
        model_names.RCNNAttZYang,
        model_names.RCNNAttPZhou
    ]


def create_ds_ctx_model_handlers():
    """ Distant supervision learning, single context based models
    """
    model_names = ModelNames()
    return {
        model_names.SelfAttentionBiLSTM: run_testing_ds_self_att_bilstm,
        model_names.AttSelfPZhouBiLSTM: run_testing_ds_att_bilstm_p_zhou,
        model_names.AttSelfZYangBiLSTM: run_testing_ds_att_hidden_zyang_bilstm,
        model_names.BiLSTM: run_testing_ds_bilstm,
        model_names.CNN: run_testing_ds_cnn,
        model_names.LSTM: run_testing_ds_lstm,
        model_names.PCNN: run_testing_ds_pcnn,
        model_names.RCNN: run_testing_ds_rcnn,
        model_names.RCNNAttZYang: run_testing_ds_rcnn_z_yang,
        model_names.RCNNAttPZhou: run_testing_ds_rcnn_p_zhou,
    }


def create_ds_mi_model_handlers():
    model_names = MaxPoolingModelNames()
    return {
        model_names.SelfAttentionBiLSTM: run_testing_ds_mi_self_att_bilstm,
        model_names.AttSelfPZhouBiLSTM: run_testing_ds_mi_att_self_p_zhou,
        model_names.AttSelfZYangBiLSTM: run_testing_ds_mi_att_self_z_yang,
        model_names.CNN: run_testing_ds_mi_cnn,
        model_names.LSTM: run_testing_ds_mi_lstm,
        model_names.PCNN: run_testing_ds_mi_pcnn,
        model_names.RCNN: run_testing_ds_mi_rcnn,
    }


def create_sl_ctx_model_handlers():
    """ Supervised learning, single context based models
    """
    model_names = ModelNames()
    return {
        model_names.SelfAttentionBiLSTM: run_testing_self_att_bilstm,
        model_names.AttSelfPZhouBiLSTM: run_testing_att_bilstm_p_zhou,
        model_names.AttSelfZYangBiLSTM: run_testing_att_hidden_zyang_bilstm,
        model_names.BiLSTM: run_testing_bilstm,
        model_names.CNN: run_testing_cnn,
        model_names.LSTM: run_testing_lstm,
        model_names.PCNN: run_testing_pcnn,
        model_names.RCNN: run_testing_rcnn,
        model_names.RCNNAttZYang: run_testing_rcnn_z_yang,
        model_names.RCNNAttPZhou: run_testing_rcnn_p_zhou,
    }


def create_sl_mi_model_handlers():
    model_names = MaxPoolingModelNames()
    return {
        model_names.SelfAttentionBiLSTM: run_mi_testing_self_att_bilstm,
        model_names.AttSelfPZhouBiLSTM: run_mi_testing_att_bilstm_p_zhou,
        model_names.AttSelfZYangBiLSTM: run_mi_testing_att_bilstm_z_yang,
        model_names.CNN: run_mi_testing_cnn,
        model_names.LSTM: run_mi_testing_lstm,
        model_names.PCNN: run_mi_testing_pcnn,
        model_names.RCNN: run_mi_testing_rcnn,
    }


def init_handlers(exp_type, model_input_type):
    assert(isinstance(exp_type, str))
    assert(isinstance(model_input_type, str))

    # Supervised learning only.
    if exp_type == SUPERVISED_LEARNING:
        if model_input_type == INPUT_TYPE_SINGLE_INSTANCE:
            return create_sl_ctx_model_handlers()
        if model_input_type == INPUT_TYPE_MULTI_INSTANCE:
            return create_sl_mi_model_handlers()

    # Distant supervision with distant supervision.
    if exp_type == SUPERVISED_LEARNING_WITH_DS:
        if model_input_type == INPUT_TYPE_SINGLE_INSTANCE:
            return None
        if model_input_type == INPUT_TYPE_MULTI_INSTANCE:
            return None


def get_callback_func(exp_type, cv_count):
    assert(isinstance(cv_count, int))

    if exp_type == SUPERVISED_LEARNING:
        if cv_count == 1:
            return classic_common_callback_modification_func
        else:
            return classic_cv_common_callback_modification_func
    if exp_type == SUPERVISED_LEARNING_WITH_DS:
        if cv_count == 1:
            return ds_common_callback_modification_func
        else:
            return ds_cv_common_callback_modification_func


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data serializer (train/test) for further experiments organization")

    # Composing cmd arguments.
    RuAttitudesVersionArg.add_argument(parser)
    CvCountArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)
    ExperimentTypeArg.add_argument(parser)

    parser.add_argument('--model-input-type',
                        dest='model_input_type',
                        type=unicode,
                        choices=[INPUT_TYPE_SINGLE_INSTANCE,
                                 INPUT_TYPE_MULTI_INSTANCE,
                                 INPUT_TYPE_MULTI_INSTANCE_WITH_ATTENTION],
                        default='ctx',
                        nargs='?',
                        help='Input format type')

    parser.add_argument('--model-name',
                        dest='model_name',
                        type=unicode,
                        choices=supported_model_names(),
                        nargs=1,
                        help='Name of a model to be utilized in experiment')

    parser.add_argument('--load-pretrained',
                        dest='pretrained_filepath',
                        type=unicode,
                        default=None,
                        nargs='?',
                        help='Load pretrained state')

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    exp_type = ExperimentTypeArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    cv_count = CvCountArg.read_argument(args)
    ra_version = RuAttitudesVersionArg.read_argument(args)
    model_input_type = args.model_input_type
    pretrained_filepath = args.pretrained_filepath
    model_name = args.model_name

    # init handler
    ctx_sl_handlers = create_sl_ctx_model_handlers()
    callback_func = get_callback_func(exp_type=exp_type, cv_count=cv_count)
    handlers = init_handlers(exp_type=exp_type, model_input_type=model_input_type)
    handler = handlers[args.model[0]]

    # Creating experiment
    labels_scaler = Common.create_labels_scaler(labels_count)
    data_io = RuSentRelBasedExperimentsIOUtils(model_states_dir=pretrained_filepath,
                                               labels_scaler=labels_scaler)

    experiment = Common.create_experiment(exp_type=exp_type,
                                          data_io=data_io,
                                          cv_count=cv_count,
                                          model_name=model_name,
                                          ra_version=ra_version)

    handler(load_model=pretrained_filepath is not None,
            experiment=experiment,
            custom_callback_func=callback_func)
