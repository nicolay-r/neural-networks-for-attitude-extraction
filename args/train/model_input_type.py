from enum import Enum
from args.base import BaseArg


class ModelInputType(Enum):
    SingleInstance = 0
    MultiInstance = 1
    MultiInstanceWithAttention = 2


class ModelInputTypeService(object):

    __names = {
        u"ctx": ModelInputType.SingleInstance,
        u'mi': ModelInputType.MultiInstance,
        u'mi-att': ModelInputType.MultiInstanceWithAttention
    }

    @staticmethod
    def __iter_supported_names():
        return iter(ModelInputTypeService.__names.keys())

    @staticmethod
    def get_type_by_name(name):
        return ModelInputTypeService.__names[name]

    @staticmethod
    def find_name_by_type(input_type):
        assert(isinstance(input_type, ModelInputType))

        for name in ModelInputTypeService.__iter_supported_names():
            related_type = ModelInputTypeService.__names[name]
            if related_type == input_type:
                return name

    @staticmethod
    def iter_supported_names():
        return ModelInputTypeService.__iter_supported_names()


class ModelInputTypeArg(BaseArg):

    _default = ModelInputType.SingleInstance

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return ModelInputTypeService.get_type_by_name(args.input_type)

    @staticmethod
    def add_argument(parser):
        str_def = ModelInputTypeService.find_name_by_type(ModelInputTypeArg._default)
        parser.add_argument('--model-input-type',
                            dest='input_type',
                            type=unicode,
                            choices=list(ModelInputTypeService.iter_supported_names()),
                            default=str_def,
                            nargs='?',
                            help='Input format type (Default: {})'.format(str_def))
