from arekit.contrib.networks.core.io_utils import NetworkIOUtils


class CustomNetworkExperimentIO(NetworkIOUtils):

    default_sources_dir = u"output"

    def get_experiment_sources_dir(self):
        return self.default_sources_dir
