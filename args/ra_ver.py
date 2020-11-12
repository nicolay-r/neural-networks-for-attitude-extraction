from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersionsService
from args.base import BaseArg


class RuAttitudesVersionArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--ra-ver',
                            dest='ra_version',
                            type=unicode,
                            nargs='?',
                            choices=list(RuAttitudesVersionsService.iter_supported_names()),
                            default=None,
                            help='RuAttitudes collection version')

    @staticmethod
    def read_argument(args):
        if args.ra_version is None:
            return None

        return RuAttitudesVersionsService.find_by_name(args.ra_version)
