from args.base import BaseArg
from arekit.contrib.bert.entity.types import BertEntityFormattersService


class EnitityFormatterTypesArg(BaseArg):

    @staticmethod
    def read_argument(args):
        name = args.entity_fmt[0]
        return BertEntityFormattersService.get_type_by_name(name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--entity-fmt',
                            dest='entity_fmt',
                            type=unicode,
                            choices=list(BertEntityFormattersService.iter_supported_names()),
                            nargs=1,
                            help='Entity formatter type')