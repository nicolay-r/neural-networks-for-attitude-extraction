import argparse
import sys

import numpy as np
import pandas as pd
from os.path import join, exists

sys.path.append('../')

from arekit.common.experiment.folding.types import FoldingType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions



class ResultsTable:

    MODEL_NAME_COL = u'model_name'
    DS_TYPE_COL = u'ds_type'

    sl_template = u"rsr-{rsr_version}-{folding_str}-balanced-tpc50_{labels_count}l"
    sl_ds_template = u"rsr-{rsr_version}-ra-{ra_version}-{folding_str}-balanced-tpc50_{labels_count}l"
    model_name_template = u"{folding_str}_{input_type}_{model_name}"

    def __init__(self, output_dir):
        assert(isinstance(output_dir, unicode))

        # Composing table for results using dataframe.
        dtypes_list = [(self.MODEL_NAME_COL, unicode),
                       (self.DS_TYPE_COL, unicode),
                       ('f1_2_avg', float), ('f1_2_cv1', float), ('f1_2_cv2', float), ('f1_2_cv3', float),
                       ('f1_2_test', float),
                       ('f1_3_avg', float), ('f1_3_cv1', float), ('f1_3_cv2', float), ('f1_3_cv3', float),
                       ('f1_3_test', float)]
        np_data = np.empty(0, dtype=np.dtype(dtypes_list))
        self.__df = pd.DataFrame(np_data)
        self.__output_dir = output_dir

    def save(self):
        print self.__df
        self.__df.to_latex("results.tex")

    @staticmethod
    def __parse_result(filepath):
        # Example to parse:
        # F1-last avg.: 0.287
        # F1-last per iterations: [0.32, 0.229, 0.311]

        f1_last = 0
        iters = None
        with open(filepath) as f:
            for line in f.readlines():
                if u'last avg' in line:
                    f1_last = float(line.split(':')[1])
                if u'last per iterations' in line:
                    arr = line.split(':')[1].strip()
                    vals = arr[1:-1]
                    iters = [float(v) for v in vals.split(',')]

        return iters, f1_last

    def __save_results(self, it_results, avg_res, labels_count, folding_type, row_ind):

        # set avg. result.
        col_name_avg = u"f1_{labels_count}_{folding}".format(
            labels_count=labels_count,
            folding=u'avg' if folding_type == FoldingType.CrossValidation else u'test')
        self.__df.set_value(row_ind, col_name_avg, avg_res)

        # setting up it_results
        if folding_type != FoldingType.CrossValidation:
            return

        for cv_index, cv_value in enumerate(it_results):
            col_name = u"f1_{labels_count}_cv{it}".format(labels_count=labels_count,
                                                          it=cv_index+1)
            assert(col_name in self.__df.columns)
            self.__df.set_value(row_ind, col_name, cv_value)

    def __for_experiment(self, model_name, folding_type, experiment_dir, exp_type, labels_count):
        assert(isinstance(model_name, ModelNames))
        assert(isinstance(folding_type, FoldingType))
        assert(isinstance(exp_type, unicode))

        input_type = u'ctx'
        folding_str = u"cv" if folding_type == FoldingType.CrossValidation else u"fx"
        results_filepath = u"log/cb_eval_avg_test.log"
        model_str = model_name.value

        # IMPORTANT:
        # This allows us to combine neut with non-neut (for 2-scale).
        ds_col_type = exp_type.replace(u'_neut', '')

        model_dir = self.model_name_template.format(folding_str=folding_str,
                                                    input_type=input_type,
                                                    model_name=model_str)

        # if the latter results are not presented
        # then we just reject the related line from the results.
        target_file = join(self.__output_dir, experiment_dir, model_dir, results_filepath)
        # print '{}'.format(target_file)
        if not exists(target_file):
            return

        # finding the related row index in a df table.
        row_ids = self.__df.index[(self.__df[self.MODEL_NAME_COL] == model_str) &
                                  (self.__df[self.DS_TYPE_COL] == ds_col_type)].tolist()

        # Providing new row if the latter does not exist.
        if len(row_ids) == 0:
            self.__df = self.__df.append({
                self.MODEL_NAME_COL: model_str,
                self.DS_TYPE_COL: ds_col_type,
            }, ignore_index=True)
            row_ind = len(self.__df) - 1
        else:
            row_ind = row_ids[0]

        # parsing results in order to organize the result table.
        it_results, avg_res = self.__parse_result(target_file)

        # saving resutls.
        self.__save_results(it_results=it_results,
                            avg_res=avg_res,
                            folding_type=folding_type,
                            row_ind=row_ind,
                            labels_count=labels_count)

    def __for_model(self, model_name, folding_type, labels_count, ra_versions):
        assert(isinstance(model_name, ModelNames))
        assert(isinstance(folding_type, FoldingType))
        assert(labels_count == 2 or labels_count == 3)

        folding_str = u"cv3" if folding_type == FoldingType.CrossValidation else u"fixed"

        # sl template
        for enum_type in RuSentRelVersions:
            self.__for_experiment(model_name=model_name,
                                  folding_type=folding_type,
                                  exp_type=u'-',
                                  labels_count=labels_count,
                                  experiment_dir=self.sl_template.format(rsr_version=enum_type.value,
                                                                         folding_str=folding_str,
                                                                         labels_count=str(labels_count)))

            # sl-ds template
            for ra_version in ra_versions:
                assert(isinstance(ra_version, RuAttitudesVersions))
                ra_name = ra_version.value
                self. __for_experiment(model_name=model_name,
                                       folding_type=folding_type,
                                       exp_type=ra_name,
                                       labels_count=labels_count,
                                       experiment_dir=self.sl_ds_template.format(rsr_version=enum_type.value,
                                                                                 ra_version=ra_name,
                                                                                 folding_str=folding_str,
                                                                                 labels_count=str(labels_count)))

    def fill(self):
        # for 3-scale
        for model_name in ModelNames:
            for folding_type in FoldingType:
                self.__for_model(model_name=model_name,
                                 folding_type=folding_type,
                                 labels_count=3,
                                 ra_versions=[RuAttitudesVersions.V12,
                                              RuAttitudesVersions.V20BaseNeut,
                                              RuAttitudesVersions.V20LargeNeut])

        # for 2-scale
        for model_name in ModelNames:
            for folding_type in FoldingType:
                self.__for_model(model_name=model_name,
                                 folding_type=folding_type,
                                 labels_count=2,
                                 ra_versions=[RuAttitudesVersions.V12,
                                              RuAttitudesVersions.V20Base,
                                              RuAttitudesVersions.V20Large])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tex table Gen")

    parser.add_argument('--results-dir',
                        dest='output_dir',
                        type=unicode,
                        nargs='?',
                        default=None,
                        help='Results dir')

    args = parser.parse_args()

    # fp = "output/rsr-v1_1-cv3-balanced-tpc50_3l/cv_ctx_cnn/log/cb_eval_avg_test.log"
    # iters, last = __parse_and(filepath=fp)
    # print iters, last

    rt = ResultsTable(args.output_dir)
    rt.fill()
    rt.save()
