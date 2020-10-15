from args.base import BaseArg


class CvCountArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--cv-count',
                            dest='cv_count',
                            type=int,
                            nargs='?',
                            default=1,
                            help='Cross-Validation, folds count')

    @staticmethod
    def read_argument(args):
        return args.cv_count
