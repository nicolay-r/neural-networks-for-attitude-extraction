#!/usr/bin/python
import sys
sys.path.append('../../../')
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from rusentrel.mi_names import AttSelfOverInstancesModelNames
from rusentrel.classic.mi.pcnn import run_testing_pcnn

if __name__ == "__main__":

    run_testing_pcnn(
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig
    )
