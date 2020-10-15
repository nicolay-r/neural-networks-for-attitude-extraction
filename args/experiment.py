from args.base import BaseArg

# Experiment types.
SUPERVISED_LEARNING = u'sl'
SUPERVISED_LEARNING_WITH_DS = u'sl+ds'
DISTANT_SUPERVISION = u'ds'


class ExperimentTypeArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.exp_type[0]

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--experiment',
                            dest='exp_type',
                            type=unicode,
                            choices=[SUPERVISED_LEARNING,
                                     SUPERVISED_LEARNING_WITH_DS,
                                     DISTANT_SUPERVISION],
                            nargs=1,
                            help='Experiment type')
