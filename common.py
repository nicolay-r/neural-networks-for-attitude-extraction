import logging

from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.contrib.experiments.ruattitudes.experiment import RuAttitudesExperiment
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.experiments.rusentrel_ds.experiment import RuSentRelWithRuAttitudesExperiment
from args.experiment import SUPERVISED_LEARNING, SUPERVISED_LEARNING_WITH_DS, DISTANT_SUPERVISION
from rusentrel.rusentrel_ds.common import DS_NAME_PREFIX


logger = logging.getLogger(__name__)


class Common:

    CV_NAME_PREFIX = u'cv_'

    @staticmethod
    def create_full_model_name(exp_type, cv_count, model_name):
        cv_prefix = Common.CV_NAME_PREFIX if cv_count > 0 else ""  # TODO. name prefixes are: cv, ds_cv, ds, ""
        exp_prefix = DS_NAME_PREFIX if exp_type == SUPERVISED_LEARNING_WITH_DS else ""
        return u"{}{}{}".format(cv_prefix, exp_prefix, model_name)

    @staticmethod
    def create_experiment(exp_type, experiment_data, cv_count, rusentrel_version, ruattitudes_version=None):
        assert(isinstance(experiment_data, DataIO))
        assert(isinstance(cv_count, int))

        experiment_data.CVFoldingAlgorithm.set_cv_count(cv_count)

        if exp_type == SUPERVISED_LEARNING:
            # Supervised learning experiment type.
            return RuSentRelExperiment(data_io=experiment_data,
                                       version=rusentrel_version)

        if exp_type == SUPERVISED_LEARNING_WITH_DS:
            # Supervised learning with an application of distant supervision in training process.
            return RuSentRelWithRuAttitudesExperiment(ruattitudes_version=ruattitudes_version,
                                                      data_io=experiment_data,
                                                      rusentrel_version=rusentrel_version)

        if exp_type == DISTANT_SUPERVISION:
            # Application of the distant supervision only (assumes for pretraining purposes)
            return RuAttitudesExperiment(data_io=experiment_data,
                                         version=ruattitudes_version)

    @staticmethod
    def create_labels_scaler(labels_count):
        assert(isinstance(labels_count, int))

        if labels_count == 2:
            return TwoLabelScaler()
        if labels_count == 3:
            return ThreeLabelScaler()

        raise NotImplementedError("Not supported")


