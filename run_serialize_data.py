import argparse
import logging
from common import Common
from io_utils import RuSentRelBasedExperimentsIOUtils

from args.cv_index import CvCountArg
from args.experiment import ExperimentTypeArg
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.engine import ExperimentEngine


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data serializer (train/test) for further experiments organization")

    # Providing arguments.
    RuAttitudesVersionArg.add_argument(parser)
    CvCountArg.add_argument(parser)
    ExperimentTypeArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    exp_type = ExperimentTypeArg.read_argument(args)
    ra_version = RuAttitudesVersionArg.read_argument(args)
    cv_count = CvCountArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)

    # Preparing necesary structures for further initializations.
    labels_scaler = Common.create_labels_scaler(labels_count)
    data_io = RuSentRelBasedExperimentsIOUtils(labels_scaler=labels_scaler)
    config = DefaultNetworkConfig()
    experiment = Common.create_experiment(exp_type=exp_type,
                                          data_io=data_io,
                                          cv_count=cv_count,
                                          model_name=u"NONAME",
                                          ra_version=ra_version)

    # Setup logger info
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # Performing serialization process.
    ExperimentEngine.run_serialization(logger=logger,
                                       experiment=experiment,
                                       create_config=lambda: config,
                                       skip_if_folder_exists=True)
