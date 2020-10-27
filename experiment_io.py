from os.path import join, dirname

from arekit.contrib.networks.core.io_utils import NetworkIOUtils


class CustomNetworkExperimentIO(NetworkIOUtils):

    @classmethod
    def get_experiment_sources_dir(cls):
        return join(dirname(__file__), u"data/")
