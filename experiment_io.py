from arekit.contrib.networks.core.io_utils import NetworkIOUtils


class CustomNetworkExperimentIO(NetworkIOUtils):

    def get_experiment_sources_dir(self):
        return u"output"
