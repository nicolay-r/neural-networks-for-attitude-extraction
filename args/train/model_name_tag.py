from args.base import BaseArg


class ModelNameTagArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return unicode(args.model_tag)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--model-tag',
                            dest='model_tag',
                            type=unicode,
                            default=u"",
                            nargs='?',
                            help='Optional and additional custom model name suffix.')
