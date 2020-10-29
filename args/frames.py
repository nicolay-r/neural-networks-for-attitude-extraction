from arekit.contrib.source.rusentiframes.io_utils import RuSentiFramesVersions
from args.base import BaseArg


class RuSentiFramesVersionArg(BaseArg):

    supported = {
        u"1.0": RuSentiFramesVersions.V10,
        u"2.0": RuSentiFramesVersions.V20
    }

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return RuSentiFramesVersionArg.supported[args.frames_version]

    @staticmethod
    def add_argument(parser):
        default = u"2.0"
        parser.add_argument('--frames-version',
                            dest='frames_version',
                            type=unicode,
                            default=default,
                            choices=list(RuSentiFramesVersionArg.supported.iterkeys()),
                            nargs='?',
                            help='Version of RuSentiFrames collection (Default: {})'.format(default))
