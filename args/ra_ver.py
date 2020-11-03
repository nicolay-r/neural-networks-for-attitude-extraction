from args.base import BaseArg
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions


# RuAttitudes supported versions
ra_versions = {
    u'1.1': RuAttitudesVersions.V11,
    u'1.2': RuAttitudesVersions.V12,
    u'2.0b': RuAttitudesVersions.V20Base,
    u'2.0l': RuAttitudesVersions.V20Large
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
