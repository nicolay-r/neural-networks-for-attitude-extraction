from args.base import BaseArg
from args.default import EPOCHS_COUNT


class EpochsCountArg(BaseArg):

    default = EPOCHS_COUNT

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.epochs

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--epochs',
                            dest='epochs',
                            type=int,
                            default=EpochsCountArg.default,
                            nargs='?',
                            help='Epochs count (Default: {})'.format(EpochsCountArg.default))
