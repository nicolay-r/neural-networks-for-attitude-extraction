import argparse
from os.path import join

from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.experiment.folding.types import FoldingType
from arekit.contrib.experiments.factory import create_experiment
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.run_training import NetworksTrainingEngine
from args.balance import UseBalancingArg
from args.cv_index import CvCountArg
from args.default import BAG_SIZE, TEST_EVERY_K_EPOCH, EPOCHS_COUNT
from args.dist_in_terms_between_ends import DistanceInTermsBetweenAttitudeEndsArg
from args.experiment import ExperimentTypeArg
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg
from args.rusentrel import RuSentRelVersionArg
from args.stemmer import StemmerArg
from args.train_acc_limit import TrainAccuracyLimitArg
from args.train_bags_per_minibatch import BagsPerMinibatchArg
from args.train_dropout_keep_prob import DropoutKeepProbArg
from args.train_f1_limit import TrainF1LimitArg
from args.train_learning_rate import LearningRateArg
from args.train_terms_per_context import TermsPerContextArg
from callback import NeuralNetworkCustomEvaluationCallback
from common import Common
# TODO. Move this parameters into args/input_format.py
from data_training import RuSentRelTrainingData
from experiment_io import CustomNetworkExperimentIO
from factory_networks import \
    INPUT_TYPE_SINGLE_INSTANCE, \
    INPUT_TYPE_MULTI_INSTANCE, \
    INPUT_TYPE_MULTI_INSTANCE_WITH_ATTENTION, \
    compose_network_and_network_config_funcs, \
    create_bags_collection_type
from factory_config_setups import modify_config_for_model, optionally_modify_config_for_experiment
from rusentrel.ctx_names import ModelNames


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data serializer (train/test) for further experiments organization")

    # Composing cmd arguments.
    RuAttitudesVersionArg.add_argument(parser)
    CvCountArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)
    ExperimentTypeArg.add_argument(parser)
    RuSentRelVersionArg.add_argument(parser)
    StemmerArg.add_argument(parser)
    DropoutKeepProbArg.add_argument(parser)
    BagsPerMinibatchArg.add_argument(parser)
    TermsPerContextArg.add_argument(parser)
    LearningRateArg.add_argument(parser)
    DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser)
    TrainAccuracyLimitArg.add_argument(parser)
    TrainF1LimitArg.add_argument(parser)

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

    parser.add_argument('--model-state-dir',
                        dest='model_load_dir',
                        type=unicode,
                        default=None,
                        nargs='?',
                        help='Use pretrained state as initial')

    parser.add_argument('--emb-filepath',
                        dest='embedding_filepath',
                        type=unicode,
                        nargs='?',
                        help='Custom embedding filepath')

    parser.add_argument('--vocab-filepath',
                        dest='vocab_filepath',
                        type=unicode,
                        nargs='?',
                        help='Custom vocabulary filepath')

    parser.add_argument('--do-eval',
                        dest='do_eval',
                        type=unicode,
                        const=True,
                        default=False,
                        nargs='?',
                        help='Perform evaluation during training process')

    parser.add_argument('--test-every-k-epoch',
                        dest='test_every_k_epoch',
                        type=int,
                        default=TEST_EVERY_K_EPOCH,
                        nargs='?',
                        help='Denotes how much epochs should be skipped before every iteration')

    parser.add_argument('--balanced-input',
                        dest='balanced_input',
                        type=bool,
                        default=UseBalancingArg.get_default(),
                        nargs=1,
                        help='Balanced input of the Train set"'
                             '"(Default: {})'.format(UseBalancingArg.get_default()))

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    exp_type = ExperimentTypeArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    rusentrel_version = RuSentRelVersionArg.read_argument(args)
    cv_count = CvCountArg.read_argument(args)
    ra_version = RuAttitudesVersionArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)
    model_input_type = args.model_input_type
    model_load_dir = args.model_load_dir
    model_name = unicode(args.model_name[0])
    embedding_filepath = args.embedding_filepath
    vocab_filepath = args.vocab_filepath
    do_eval = args.do_eval
    dropout_keep_prob = DropoutKeepProbArg.read_argument(args)
    bags_per_minibatch = BagsPerMinibatchArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    learning_rate = LearningRateArg.read_argument(args)
    test_every_k_epoch = args.test_every_k_epoch
    balanced_input = args.balanced_input
    dist_in_terms_between_attitude_ends = DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)
    train_acc_limit = TrainAccuracyLimitArg.read_argument(args)
    train_f1_limit = TrainF1LimitArg.read_argument(args)

    # Defining folding type
    folding_type = FoldingType.Fixed if cv_count == 1 else FoldingType.CrossValidation

    # init handler
    bags_collection_type = create_bags_collection_type(model_input_type=model_input_type)
    network_func, network_config_func = compose_network_and_network_config_funcs(
        model_name=model_name,
        model_input_type=model_input_type)

    #####################
    # Initialize callback
    #####################
    callback = NeuralNetworkCustomEvaluationCallback(do_eval=do_eval,
                                                     train_acc_limit=train_acc_limit,
                                                     train_f1_limit=train_f1_limit)

    # We keep parameters only for fixed experiment type by default.
    callback.set_key_save_hidden_parameters(folding_type == FoldingType.Fixed)
    # We stop training process according to the present at some prior
    # cost values in case of experiments with cv-based doc-ids folding format.
    callback.set_key_stop_training_by_cost(folding_type == FoldingType.CrossValidation)
    # We use a predefined value for total amount of epochs and for evaluation iterations.
    callback.set_test_on_epochs(range(0, EPOCHS_COUNT + 1, test_every_k_epoch))

    # Creating experiment
    evaluator = TwoClassEvaluator()
    experiment_data = RuSentRelTrainingData(
        labels_scaler=Common.create_labels_scaler(labels_count),
        stemmer=stemmer,
        opinion_formatter=Common.create_opinion_collection_formatter(),
        evaluator=evaluator,
        callback=callback)

    extra_name_suffix = Common.create_exp_name_suffix(
        use_balancing=balanced_input,
        terms_per_context=terms_per_context,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    experiment = create_experiment(exp_type=exp_type,
                                   experiment_data=experiment_data,
                                   folding_type=folding_type,
                                   rusentrel_version=rusentrel_version,
                                   ruattitudes_version=ra_version,
                                   experiment_io_type=CustomNetworkExperimentIO,
                                   extra_name_suffix=extra_name_suffix,
                                   load_ruattitude_docs=False)

    full_model_name = Common.create_full_model_name(folding_type=folding_type,
                                                    model_name=model_name)

    model_io = NeuralNetworkModelIO(full_model_name=full_model_name,
                                    target_dir=experiment.ExperimentIO.get_target_dir(),
                                    source_dir=model_load_dir,
                                    embedding_filepath=embedding_filepath,
                                    vocab_filepath=vocab_filepath)

    # Setup logging dir.
    callback.set_log_dir(join(model_io.get_model_dir(), u"log/"))
    # Setup model io.
    experiment_data.set_model_io(model_io)

    ###################
    # Initialize config
    ###################
    config = network_config_func()

    assert(isinstance(config, DefaultNetworkConfig))

    # Default settings, applied from cmd arguments.
    config.modify_classes_count(value=experiment.DataIO.LabelsScaler.classes_count())
    config.modify_learning_rate(learning_rate)
    config.modify_use_class_weights(True)
    config.modify_dropout_keep_prob(dropout_keep_prob)
    config.modify_bag_size(BAG_SIZE)
    config.modify_bags_per_minibatch(bags_per_minibatch)
    config.modify_embedding_dropout_keep_prob(1.0)
    config.modify_terms_per_context(terms_per_context)
    config.modify_use_entity_types_in_embedding(False)

    # Modify config parameters. This may affect
    # the settings, already applied above!
    optionally_modify_config_for_experiment(exp_type=exp_type,
                                            model_input_type=model_input_type,
                                            config=config)

    # Modify config parameters. This may affect
    # the settings, already applied above!
    modify_config_for_model(model_name=model_name,
                            model_input_type=model_input_type,
                            config=config)

    training_engine = NetworksTrainingEngine(load_model=model_load_dir is not None,
                                             experiment=experiment,
                                             create_network_func=network_func,
                                             config=config,
                                             bags_collection_type=bags_collection_type)

    training_engine.run()
