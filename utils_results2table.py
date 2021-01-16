import argparse
import sys

import numpy as np
import pandas as pd
from os.path import join, exists


sys.path.append('../')

from arekit.common.experiment.folding.types import FoldingType
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions, RuAttitudesVersionsService
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions


class ResultsTable(object):

    MODEL_NAME_COL = u'model_name'
    DS_TYPE_COL = u'ds_type'

    sl_template = u"rsr-{rsr_version}-{folding_str}-balanced-tpc50_{labels_count}l"
    sl_ds_template = u"rsr-{rsr_version}-ra-{ra_version}-{folding_str}-balanced-tpc50_{labels_count}l"
    __model_name_template = u"{folding_str}_{input_type}_{model_name}"

    def __init__(self, input_type, output_dir):
        assert(isinstance(input_type, unicode))
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
        self.__input_type = input_type
        self.__output_dir = output_dir

    def _create_output_basename(self):
        return "results-{input_type}".format(input_type=self.__input_type)

    def save(self):
        basename = self._create_output_basename()
        self.__df.to_latex(basename + '.tex', na_rep='', index=False)

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

    def _create_model_dir(self, folding_str, model_str, exp_type_name):
        return self.__model_name_template.format(folding_str=folding_str,
                                                 input_type=self.__input_type,
                                                 model_name=model_str)

    def _for_experiment(self, model_name, folding_type, experiment_dir, ra_type, labels_count):
        assert(isinstance(model_name, ModelNames))
        assert(isinstance(folding_type, FoldingType))
        assert(isinstance(ra_type, RuAttitudesVersions) or ra_type is None)

        exp_type_name = ra_type.value if isinstance(ra_type, RuAttitudesVersions) else u'-'
        folding_str = u"cv" if folding_type == FoldingType.CrossValidation else u"fx"
        results_filepath = u"log/cb_eval_avg_test.log"
        model_str = model_name.value

        # IMPORTANT:
        # This allows us to combine neut with non-neut (for 2-scale).
        ds_col_type = exp_type_name.replace(u'_neut', '')

        model_dir = self._create_model_dir(folding_str=folding_str,
                                           model_str=model_str,
                                           exp_type_name=exp_type_name)

        # if the latter results are not presented
        # then we just reject the related line from the results.
        target_file = join(self.__output_dir, experiment_dir, model_dir, results_filepath)
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

        # saving results.
        self.__save_results(it_results=it_results,
                            avg_res=avg_res,
                            folding_type=folding_type,
                            row_ind=row_ind,
                            labels_count=labels_count)

    def register(self, model_name, folding_type, labels_count, ra_version):
        assert(isinstance(model_name, ModelNames))
        assert(isinstance(folding_type, FoldingType))
        assert(isinstance(ra_version, RuAttitudesVersions) or ra_version is None)
        assert(labels_count == 2 or labels_count == 3)

        folding_str = u"cv3" if folding_type == FoldingType.CrossValidation else u"fixed"
        rsr_version = RuSentRelVersions.V11

        if ra_version is None:
            # Using supervised learning only
            ra_type = None
            exp_dir = self.sl_template.format(rsr_version=rsr_version.value,
                                              folding_str=folding_str,
                                              labels_count=str(labels_count))
        else:
            # Using distant supervision in combination with supervised learning.
            ra_type = ra_version
            exp_dir = self.sl_ds_template.format(rsr_version=rsr_version.value,
                                                 ra_version=ra_version.value,
                                                 folding_str=folding_str,
                                                 labels_count=str(labels_count))

        self._for_experiment(model_name=model_name,
                             folding_type=folding_type,
                             ra_type=ra_type,
                             labels_count=labels_count,
                             experiment_dir=exp_dir)


class FineTunedResultsProvider(ResultsTable):

    __fine_tuned_suffix = u"{model_name}-ft-{model_tag}"

    __tags = {
        RuAttitudesVersions.V12: u'ra12',
        RuAttitudesVersions.V20Base: u'ra20b',
        RuAttitudesVersions.V20BaseNeut: u'ra20bn',
        RuAttitudesVersions.V20Large: u'ra20l',
        RuAttitudesVersions.V20LargeNeut: u'ra20ln'
    }

    def _create_output_basename(self):
        base_name = super(FineTunedResultsProvider, self)._create_output_basename()
        return base_name + u'-ft'

    @staticmethod
    def __model_tag_from_ra_version(ra_version):
        assert(isinstance(ra_version, RuAttitudesVersions))
        return FineTunedResultsProvider.__tags[ra_version]

    def _create_model_dir(self, folding_str, model_str, exp_type_name):
        origin_name = super(FineTunedResultsProvider, self)._create_model_dir(
            folding_str=folding_str,
            model_str=model_str,
            exp_type_name=exp_type_name)

        ra_version = RuAttitudesVersionsService.find_by_name(exp_type_name)
        updated_name = self.__fine_tuned_suffix.format(model_name=origin_name,
                                                       model_tag=self.__model_tag_from_ra_version(ra_version))

        return updated_name

    def _for_experiment(self, model_name, folding_type, experiment_dir, ra_type, labels_count):
        assert(ra_type is None)

        # For every tag key we gathering results
        # within a single experiment dir.
        for ra_version in self.__tags.iterkeys():
            super(FineTunedResultsProvider, self)._for_experiment(
                model_name=model_name,
                folding_type=folding_type,
                experiment_dir=experiment_dir,
                ra_type=ra_version,
                labels_count=labels_count)


def fill_single(res):
    assert(isinstance(res, ResultsTable))
    for model_name in ModelNames:
        for folding_type in FoldingType:
            # for 3-scale
            for ra_version in [RuAttitudesVersions.V20LargeNeut,
                               RuAttitudesVersions.V20BaseNeut,
                               RuAttitudesVersions.V12,
                               None]:
                res.register(model_name=model_name,
                             folding_type=folding_type,
                             labels_count=3,
                             ra_version=ra_version)

            # for 2-scale
            for ra_version in [RuAttitudesVersions.V20Large,
                               RuAttitudesVersions.V20Base,
                               RuAttitudesVersions.V12,
                               None]:
                res.register(model_name=model_name,
                             folding_type=folding_type,
                             labels_count=2,
                             ra_version=ra_version)


def fill_finetunned(res):
    assert(isinstance(res, FineTunedResultsProvider))
    for model_name in ModelNames:
        for folding_type in FoldingType:
            for labels in [2, 3]:
                res.register(model_name=model_name,
                             folding_type=folding_type,
                             labels_count=labels,
                             ra_version=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tex table Gen")

    parser.add_argument('--results-dir',
                        dest='output_dir',
                        type=unicode,
                        nargs='?',
                        default=None,
                        help='Results dir')

    parser.add_argument('--input-type',
                        dest='input_type',
                        type=unicode,
                        nargs='?',
                        default=ModelInputType.SingleInstance.value,
                        choices=[ModelInputType.SingleInstance.value,
                                 ModelInputType.MultiInstanceMaxPooling.value,
                                 ModelInputType.MultiInstanceWithSelfAttention.value],
                        help='input type format')

    parser.add_argument('--training-type',
                        dest='training_type',
                        type=unicode,
                        nargs=1,
                        default=u'single',
                        choices=[u'single', u'ft'],
                        help='Training format used for results gathering')

    args = parser.parse_args()

    training_type = args.training_type[0]

    rt = None
    if training_type == u'single':
        # Single training format.
        rt = ResultsTable(output_dir=args.output_dir,
                          input_type=args.input_type)
        fill_single(rt)
    elif training_type == u'ft':
        # Fine-tuned results format.
        rt = FineTunedResultsProvider(output_dir=args.output_dir,
                                      input_type=args.input_type)
        fill_finetunned(rt)
    else:
        raise NotImplementedError()

    rt.save()
