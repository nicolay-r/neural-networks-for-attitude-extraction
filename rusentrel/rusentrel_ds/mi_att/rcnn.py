#!/usr/bin/python
import sys


sys.path.append('../../../')
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from rusentrel.mi_names import AttSelfOverInstancesModelNames
from rusentrel.rusentrel_ds.mi.rcnn import run_testing_mi_rcnn

if __name__ == "__main__":

    run_testing_mi_rcnn(
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig
    )
