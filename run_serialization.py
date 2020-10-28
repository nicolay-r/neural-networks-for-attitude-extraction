import argparse

from common import Common

from arekit.contrib.networks.run_serializer import NetworksExperimentInputSerializer
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from serialization_data import RuSentRelExperimentSerializationData

from args.cv_index import CvCountArg
from args.experiment import ExperimentTypeArg
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg


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
    experiment_data = RuSentRelExperimentSerializationData(labels_scaler=labels_scaler)
    experiment = Common.create_experiment(exp_type=exp_type,
                                          experiment_data=experiment_data,
                                          cv_count=cv_count,
                                          rusentrel_version=RuSentRelVersions.V11,
                                          ruattitudes_version=ra_version)

    # Initialize cv_count and setup cv-splitter
    splitter = Common.create_folding_splitter(doc_operations=experiment.DocumentOperations,
                                              data_dir=experiment.ExperimentIO.get_target_dir())
    experiment_data.CVFoldingAlgorithm.set_cv_count(cv_count)
    experiment_data.CVFoldingAlgorithm.set_splitter(splitter)

    # Performing serialization process.
    serialization_engine = NetworksExperimentInputSerializer(experiment=experiment,
                                                             skip_folder_if_exists=True)

    serialization_engine.run()
