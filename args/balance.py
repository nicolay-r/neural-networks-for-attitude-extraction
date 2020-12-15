from args.base import BaseArg


class UseBalancingArg(BaseArg):

    __default = True

    def __init__(self):
        pass

    @staticmethod
    def get_default():
        return UseBalancingArg.__default

    @staticmethod
    def read_argument(args):
        return args.balance_samples[0]

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--balance-samples',
                            dest='balance_samples',
                            type=bool,
                            nargs=1,
                            help='Use balancing for Train type during sample serialization process "'
                                 '"(Default: {})'.format(UseBalancingArg.__default))
