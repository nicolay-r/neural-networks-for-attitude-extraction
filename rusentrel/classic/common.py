from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from args.default import BAGS_PER_MINIBATCH
from rusentrel.default import MI_CONTEXTS_PER_OPINION


def apply_classic_mi_settings(config):
    """
    Multi instance version
    """
    assert(isinstance(config, BaseMultiInstanceConfig))
    config.set_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)
    config.modify_bags_per_minibatch(BAGS_PER_MINIBATCH)
