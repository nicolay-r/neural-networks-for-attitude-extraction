from args.base import BaseArg
from args.default import DROPOUT_KEEP_PROB


class DropoutKeepProbArg(BaseArg):

    default = DROPOUT_KEEP_PROB

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.dropout_keep_prob

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--dropout-keep-prob',
                            dest='dropout_keep_prob',
                            type=int,
                            default=DropoutKeepProbArg.default,
                            nargs='?',
                            help='Dropout keep prob (Default: {})'.format(DropoutKeepProbArg.default))
