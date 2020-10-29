from args.base import BaseArg
from arekit.contrib.networks.entities.str_emb_fmt import StringWordEmbeddingEntityFormatter
from arekit.contrib.networks.entities.str_fmt import StringSimpleMaskedEntityFormatter


class EnitityFormatterTypesArg(BaseArg):

    default = u'simple-masked'

    supported_types = {
        u"simple-masked": StringSimpleMaskedEntityFormatter,
        u'we-masked': StringWordEmbeddingEntityFormatter,
    }

    @staticmethod
    def read_argument(args):
        fmt_type = args.entity_fmt
        return EnitityFormatterTypesArg.supported_types[fmt_type]()

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--entity-fmt',
                            dest='entity_fmt',
                            type=unicode,
                            choices=list(EnitityFormatterTypesArg.supported_types.iterkeys()),
                            default=EnitityFormatterTypesArg.default,
                            nargs='?',
                            help='Entity formatter type (Default: {})'.format(
                                EnitityFormatterTypesArg.default))
