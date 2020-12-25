from args.base import BaseArg


def supported_model_names():
    model_names = ModelNames()
    return [
        model_names.SelfAttentionBiLSTM,
        model_names.AttSelfPZhouBiLSTM,
        model_names.AttSelfZYangBiLSTM,
        model_names.BiLSTM,
        model_names.CNN,
        model_names.LSTM,
        model_names.PCNN,
        model_names.RCNN,
        model_names.RCNNAttZYang,
        model_names.RCNNAttPZhou
    ]


class ModelNameArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return unicode(args.model_name[0])

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--model-name',
                            dest='model_name',
                            type=unicode,
                            choices=supported_model_names(),
                            nargs=1,
                            help='Name of a model to be utilized in experiment')


class ModelNames(object):

    __full_name_template = u'{}{}'

    # region list of all existed models

    @property
    def CNN(self):
        return self.__full_name_template.format(self.Prefix, u'cnn')

    @property
    def AttEndsCNN(self):
        return self.__full_name_template.format(self.Prefix, u'att-cnn')

    @property
    def AttEndsAndFramesCNN(self):
        return self.__full_name_template.format(self.Prefix, u'att-ef-cnn')

    @property
    def AttSynonymEndsCNN(self):
        return self.__full_name_template.format(self.Prefix, u'att-se-cnn')

    @property
    def AttSynonymEndsPCNN(self):
        return self.__full_name_template.format(self.Prefix, u'att-se-pcnn')

    @property
    def AttSynonymEndsBiLSTM(self):
        return self.__full_name_template.format(self.Prefix, u'att-se-bilstm')

    @property
    def AttSynonymEndsAndFramesCNN(self):
        return self.__full_name_template.format(self.Prefix, u'att-sef-cnn')

    @property
    def AttSynonymEndsAndFramesPCNN(self):
        return self.__full_name_template.format(self.Prefix, u'att-sef-pcnn')

    @property
    def AttSynonymEndsAndFramesBiLSTM(self):
        return self.__full_name_template.format(self.Prefix, u'att-sef-bilstm')

    @property
    def AttEndsAndFramesPCNN(self):
        return self.__full_name_template.format(self.Prefix, u'att-ef-pcnn')

    @property
    def AttEndsAndFramesBiLSTM(self):
        return self.__full_name_template.format(self.Prefix, u'att-ef-bilstm')

    @property
    def AttEndsPCNN(self):
        return self.__full_name_template.format(self.Prefix, u'att-pcnn')

    @property
    def AttFramesCNN(self):
        return self.__full_name_template.format(self.Prefix, u'att-frames-cnn')

    @property
    def AttFramesPCNN(self):
        return self.__full_name_template.format(self.Prefix, u'att-frames-pcnn')

    @property
    def SelfAttentionBiLSTM(self):
        return self.__full_name_template.format(self.Prefix, u'self-att-bilstm')

    @property
    def BiLSTM(self):
        return self.__full_name_template.format(self.Prefix, u'bilstm')

    @property
    def IANFrames(self):
        return self.__full_name_template.format(self.Prefix, u'ian')

    @property
    def IANEnds(self):
        return self.__full_name_template.format(self.Prefix, u'ian-ends')

    @property
    def IANEndsAndFrames(self):
        return self.__full_name_template.format(self.Prefix, u'ian-ef')

    @property
    def IANSynonymEnds(self):
        return self.__full_name_template.format(self.Prefix, u'ian-se')

    @property
    def IANSynonymEndsAndFrames(self):
        return self.__full_name_template.format(self.Prefix, u'ian-sef')

    @property
    def PCNN(self):
        return self.__full_name_template.format(self.Prefix, u'pcnn')

    @property
    def LSTM(self):
        return self.__full_name_template.format(self.Prefix, u'rnn')

    @property
    def RCNN(self):
        return self.__full_name_template.format(self.Prefix, u'rcnn')

    @property
    def RCNNAttPZhou(self):
        return self.__full_name_template.format(self.Prefix, u'rcnn-att-p-zhou')

    @property
    def RCNNAttZYang(self):
        return self.__full_name_template.format(self.Prefix, u'rcnn-att-z-yang')

    @property
    def AttFramesBiLSTM(self):
        return self.__full_name_template.format(self.Prefix, u'att-frames-bilstm')

    @property
    def AttSelfZYangBiLSTM(self):
        return self.__full_name_template.format(self.Prefix, u'att-bilstm-z-yang')

    @property
    def AttSelfPZhouBiLSTM(self):
        return self.__full_name_template.format(self.Prefix, u'att-bilstm')

    # endregion

    # TODO. Remove this property.
    # TODO. Use prefix later, as the related parameter presented in cmd args.
    @property
    def Prefix(self):
        return u''