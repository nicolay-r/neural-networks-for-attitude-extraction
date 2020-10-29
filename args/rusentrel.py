from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from args.base import BaseArg


class RuSentRelVersionArg(BaseArg):

    default = RuSentRelVersions.V11

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.rsr_version

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--rsr-version',
                            dest='rsr_version',
                            type=unicode,
                            choices=[RuSentRelVersions.V11],
                            default=RuSentRelVersionArg.default,
                            nargs='?',
                            help='RuSentRel version (Default: {})'.format(
                                RuSentRelVersionArg.default))
