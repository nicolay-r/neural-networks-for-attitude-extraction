import logging
from arekit.common.experiment.data.base import DataIO

logger = logging.getLogger(__name__)


def data_io_post_initialization(experiment_data, full_model_name, cv_count):
    assert(isinstance(experiment_data, DataIO))

    logger.info("Full-Model-Name: {}".format(full_model_name))

    experiment_data.CVFoldingAlgorithm.set_cv_count(cv_count)
    experiment_data.set_model_name(full_model_name)
    experiment_data.ModelIO.set_model_name(value=full_model_name)

