from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.contrib.experiments.ruattitudes.experiment import RuAttitudesExperiment
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.experiments.rusentrel_ds.experiment import RuSentRelWithRuAttitudesExperiment
from args.experiment import SUPERVISED_LEARNING, SUPERVISED_LEARNING_WITH_DS, DISTANT_SUPERVISION
from io_utils import RuSentRelBasedExperimentsIOUtils
from rusentrel.base import data_io_post_initialization
from rusentrel.rusentrel_ds.common import DS_NAME_PREFIX


class Common:

    CV_NAME_PREFIX = u'cv_'

    @staticmethod
    def create_experiment(exp_type, data_io, cv_count, model_name, ra_version=None):
        assert(isinstance(data_io, RuSentRelBasedExperimentsIOUtils))
        assert(isinstance(cv_count, int))

        cv_prefix = Common.CV_NAME_PREFIX if cv_count > 0 else ""  # TODO. name prefixes are: cv, ds_cv, ds, ""
        exp_prefix = DS_NAME_PREFIX if exp_type == SUPERVISED_LEARNING_WITH_DS else ""
        full_model_name = u"{}{}{}".format(cv_prefix, exp_prefix, model_name)

        # Peforming post initialization
        data_io_post_initialization(data_io=data_io,
                                    full_model_name=full_model_name,
                                    cv_count=cv_count)

        if exp_type == SUPERVISED_LEARNING:
            # Supervised learning experiment type.
            return RuSentRelExperiment(data_io=data_io,
                                       version=data_io.RuSentRelVersion,
                                       prepare_model_root=True)

        if exp_type == SUPERVISED_LEARNING_WITH_DS:
            # Supervised learning with an application of distant supervision in training process.
            return RuSentRelWithRuAttitudesExperiment(version=ra_version,
                                                      data_io=data_io,
                                                      rusentrel_version=data_io.RuSentRelVersion,
                                                      prepare_model_root=True)

        if exp_type == DISTANT_SUPERVISION:
            # Application of the distant supervision only (assumes for pretraining purposes)
            return RuAttitudesExperiment(data_io=data_io,
                                         version=ra_version,
                                         prepare_model_root=True)

    @staticmethod
    def create_labels_scaler(labels_count):
        assert(isinstance(labels_count, int))

        if labels_count == 2:
            return TwoLabelScaler()
        if labels_count == 3:
            return ThreeLabelScaler()

        raise NotImplementedError("Not supported")


