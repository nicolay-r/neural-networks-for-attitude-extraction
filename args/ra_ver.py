from args.base import BaseArg
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions


# RuAttitudes supported verisons
ra_versions = {
    '1.1': RuAttitudesVersions.V11,
    '2.0': RuAttitudesVersions.V20
}


class RuAttitudesVersionArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--ra-ver',
                            dest='ra_version',
                            type=unicode,
                            nargs='?',
                            choices=list(ra_versions.iterkeys()),
                            default=False,
                            help='RuAttitudes collection version')

    @staticmethod
    def read_argument(args):
        value = args.ra_version
        return ra_versions[value] if value is not None else None
