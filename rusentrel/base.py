import logging

from arekit.common.experiment.data_io import DataIO


logger = logging.getLogger(__name__)


def data_io_post_initialization(data_io, full_model_name, cv_count):
    assert(isinstance(data_io, DataIO))

    logger.info("Full-Model-Name: {}".format(full_model_name))

    data_io.CVFoldingAlgorithm.set_cv_count(cv_count)
    data_io.set_model_name(full_model_name)
    data_io.ModelIO.set_model_name(value=full_model_name)

