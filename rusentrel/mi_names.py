from args.train.model_name import ModelNames


class MaxPoolingModelNames(ModelNames):

    @property
    def Prefix(self):
        return u'mi-'


class AttSelfOverInstancesModelNames(ModelNames):

    @property
    def Prefix(self):
        return u'miah-'


class MLPOverInstancesModelNames(ModelNames):

    @property
    def Prefix(self):
        return u'mimlp-'
