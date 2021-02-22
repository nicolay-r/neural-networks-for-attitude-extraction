#############################################################
# This script allows to export all the model results into
# tar.gz archive, rejecting non required for further
# evaluation files.
#############################################################
#!/bin/bash
tar -zcvf nn-results.tar.gz -X res_exclude_exclude.txt output
