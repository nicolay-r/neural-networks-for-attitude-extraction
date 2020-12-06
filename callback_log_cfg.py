from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig


def write_config_setups(config, out_filepath):
    assert(isinstance(config, DefaultNetworkConfig))
    with open(out_filepath, 'w') as f:
        for param in config.get_parameters():
            assert (isinstance(param, list))
            f.write(u"{}\n".format(str(param)))
