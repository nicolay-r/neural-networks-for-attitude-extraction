#!/usr/bin/python
import sys


sys.path.append('../../../')
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from rusentrel.mi_names import AttSelfOverInstancesModelNames
from rusentrel.rusentrel_ds.mi.att_self_z_yang import run_testing_mi_att_hidden_zyang


if __name__ == "__main__":

    run_testing_mi_att_hidden_zyang(
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig
    )
