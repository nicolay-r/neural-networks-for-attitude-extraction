import argparse
from tabulate import tabulate

import numpy as np
import pandas as pd
from os.path import join, exists
from enum import Enum

from arekit.common.evaluation.results.two_class import TwoClassEvalResult
from arekit.common.experiment.data_type import DataType
from arekit.contrib.experiments.rusentrel.folding import DEFAULT_CV_COUNT
from args.train.model_input_type import ModelInputTypeArg
from arekit.common.experiment.folding.types import FoldingType
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions, RuAttitudesVersionsService
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from callback_log_exp import parse_last_epoch_results
from callback_log_iter import parse_last
from callback_log_training import extract_avg_epoch_time_from_training_log, extract_last_param_value_from_training_log, \
    chop_microseconds, parse_float_network_parameter, AVG_FIT_ACC_ARGUMENT, EPOCH_ARGUMENT
from common import Common
from experiment_io import CustomNetworkExperimentIO


class ResultType(Enum):

    F1 = u'f1'
    TrainingEpochTime = u'train-time'
    TrainingTotalTime = u'train-total-time'
    TrainingAccuracy = u'train-acc'
    F1Train = u'f1-last-train'
    EpochsCount = u'epochs'

    LearningRate = u'train-lr'

    @staticmethod
    def from_str(value):
        for t in ResultType:
            if t.value == value:
                return t


class ResultsTable(object):

    MODEL_NAME_COL = u'model_name'
    DS_TYPE_COL = u'ds_type'

    sl_template = u"rsr-{rsr_version}-{folding_str}-balanced-tpc50_{labels_count}l"
    sl_ds_template = u"rsr-{rsr_version}-ra-{ra_version}-{folding_str}-balanced-tpc50_{labels_count}l"

    def __init__(self, input_type, output_dir, result_type, labels, cv_count, foldings):
        assert(isinstance(input_type, ModelInputType))
        assert(isinstance(output_dir, unicode))
        assert(isinstance(result_type, ResultType))
        assert(isinstance(cv_count, int))
        assert(isinstance(labels, list))
        assert(isinstance(foldings, list))

        def __it_columns():
            # Providing extra columns for results.
            for label in labels:
                if provide_cv_columns:
                    # cv-based results columns.
                    yield (self.__rcol_agg(label, FoldingType.CrossValidation), float)
                    for cv_index in range(cv_count):
                        yield (self.__rcol_it(label, cv_index), float)
                if provide_fx_columns:
                    # fx-based result columns.
                    yield (self.__rcol_agg(label, FoldingType.Fixed), float)

        provide_cv_columns = FoldingType.CrossValidation in foldings
        provide_fx_columns = FoldingType.Fixed in foldings

        # Composing table for results using dataframe.
        dtypes_list = [(self.MODEL_NAME_COL, unicode),
                       (self.DS_TYPE_COL, unicode)]

        # Providing result columns.
        for col in __it_columns():
            dtypes_list.append(col)

        np_data = np.empty(0, dtype=np.dtype(dtypes_list))

        self.__df = pd.DataFrame(np_data)
        self.__input_type = input_type
        self.__output_dir = output_dir
        self.__result_type = result_type
        self.__cv_count = cv_count

    @staticmethod
    def __rcol_agg(labels_count, folding_type):
        assert(isinstance(folding_type, FoldingType))
        return u"r_{labels_count}_{folding}".format(
            labels_count=labels_count,
            folding=u'avg' if folding_type == FoldingType.CrossValidation else u'test')

    @staticmethod
    def __rcol_it(labels_count, it):
        return u"r_{labels_count}_cv{it}".format(labels_count=labels_count, it=it)

    def _create_output_basename(self):
        return u"results-{input_type}-{result_type}".format(
            input_type=self.__input_type.value,
            result_type=self.__result_type.value)

    def _create_model_dir(self, folding_type, model_name, exp_type_name):
        return Common.create_full_model_name(folding_type=folding_type,
                                             input_type=self.__input_type,
                                             model_name=model_name)

    def __save_results(self, it_results, avg_res, labels_count, folding_type, row_ind):
        assert(isinstance(it_results, list))

        # set avg. result.
        col_name_avg = self.__rcol_agg(labels_count=labels_count,
                                       folding_type=folding_type)
        if col_name_avg in self.__df.columns:
            self.__df.set_value(row_ind, col_name_avg, avg_res)

        # setting up it_results
        if folding_type != FoldingType.CrossValidation:
            return

        # setup cv values.
        for cv_index, cv_value in enumerate(it_results):
            col_name = self.__rcol_it(labels_count=labels_count, it=cv_index)
            if col_name not in self.__df.columns:
                continue
            self.__df.set_value(row_ind, col_name, cv_value)

    def __iter_files_per_iteration(self, folding_type):
        assert(isinstance(folding_type, FoldingType))

        iters = self.__cv_count if folding_type == FoldingType.CrossValidation else 1

        if self.__result_type == ResultType.F1:
            yield join(Common.log_dir, Common.log_test_eval_exp_filename)
        elif self.__result_type == ResultType.TrainingEpochTime or \
                self.__result_type == ResultType.TrainingTotalTime or \
                self.__result_type == ResultType.TrainingAccuracy or \
                self.__result_type == ResultType.EpochsCount:
            for it_index in range(iters):
                yield join(Common.log_dir, Common.create_log_train_filename(data_type=DataType.Train,
                                                                            iter_index=it_index))
        elif self.__result_type == ResultType.F1Train:
            for it_index in range(iters):
                yield join(Common.log_dir, Common.create_log_eval_filename(data_type=DataType.Train,
                                                                           iter_index=it_index))
        elif self.__result_type == ResultType.LearningRate:
            for it_index in range(iters):
                yield join(Common.log_dir, Common.model_config_name)
        else:
            raise NotImplementedError("Not supported type: {}".format(self.__result_type))

    @staticmethod
    def __parse_iter_and_avg_result(r_type, files_per_iter):
        assert(isinstance(r_type, ResultType))
        assert(isinstance(files_per_iter, list))

        # parsing results in order to organize the result table.
        if r_type == ResultType.F1:
            # This is a single file which gathers
            # information across all the iterations.
            assert(len(files_per_iter) == 1)
            return parse_last_epoch_results(files_per_iter[0])
        elif r_type == ResultType.TrainingEpochTime:
            return [extract_avg_epoch_time_from_training_log(fp) for fp in files_per_iter]
        elif r_type == ResultType.TrainingAccuracy:
            return [extract_last_param_value_from_training_log(fp, key=AVG_FIT_ACC_ARGUMENT) for fp in files_per_iter]
        elif r_type == ResultType.EpochsCount:
            return [extract_last_param_value_from_training_log(fp, key=EPOCH_ARGUMENT) for fp in files_per_iter]
        elif r_type == ResultType.TrainingTotalTime:
            epochs = ResultsTable.__parse_iter_and_avg_result(r_type=ResultType.EpochsCount,
                                                              files_per_iter=files_per_iter)
            times = ResultsTable.__parse_iter_and_avg_result(r_type=ResultType.TrainingEpochTime,
                                                             files_per_iter=files_per_iter)
            return [epochs[i] * times[i] for i in range(len(epochs))]
        elif r_type == ResultType.F1Train:
            return [parse_last(filepath=fp, col=TwoClassEvalResult.C_F1) for fp in files_per_iter]
        elif r_type == ResultType.LearningRate:
            return [parse_float_network_parameter(fp, u'learning_rate') for fp in files_per_iter]
        else:
            raise NotImplementedError("Not supported type: {}". format(r_type))

    def __calc_avg_it_res(self, it_results):
        if len(it_results) == 0:
            return 0

        if len(it_results) == 1:
            return it_results[0]

        if self.__result_type == ResultType.TrainingEpochTime or \
            self.__result_type == ResultType.TrainingTotalTime:
            # These complex implementation due to datetime results
            # the latter won't work with np.mean.
            total_result = it_results[0]
            for ind in range(1, len(it_results)):
                total_result += it_results[ind]
            return chop_microseconds(total_result / len(it_results))
        else:
            return np.mean(it_results)

    def __add_row(self, exp_type_name, model_dir):
        # IMPORTANT:
        # This allows us to combine neut with non-neut (for 2-scale).
        ds_type_name = exp_type_name.replace(u'_neut', '')
        # IMPORTANT:
        # Removing first (folding-type) prefix
        model_str = model_dir[model_dir.index(u'_')+1:]

        # finding the related row index in a df table.
        row_ids = self.__df.index[(self.__df[self.MODEL_NAME_COL] == model_str) &
                                  (self.__df[self.DS_TYPE_COL] == ds_type_name)].tolist()

        # Providing new row if the latter does not exist.
        if len(row_ids) == 0:
            self.__df = self.__df.append({
                self.MODEL_NAME_COL: model_str,
                self.DS_TYPE_COL: ds_type_name,
            }, ignore_index=True)
            row_ind = len(self.__df) - 1
        else:
            row_ind = row_ids[0]

        return row_ind

    def _for_experiment(self, model_name, folding_type, experiment_dir, ra_type, labels_count):
        assert(isinstance(model_name, ModelNames))
        assert(isinstance(folding_type, FoldingType))
        assert(isinstance(ra_type, RuAttitudesVersions) or ra_type is None)

        exp_type_name = ra_type.value if isinstance(ra_type, RuAttitudesVersions) else u'-'
        model_dir = self._create_model_dir(folding_type=folding_type,
                                           model_name=model_name,
                                           exp_type_name=exp_type_name)

        # if the latter results are not presented
        # then we just reject the related line from the results.
        files_per_iter = [join(self.__output_dir, experiment_dir, model_dir, target) for
                          target in self.__iter_files_per_iteration(folding_type)]

        # Check files existance.
        for target_file in files_per_iter:
            if not exists(target_file):
                return

        it_results = self.__parse_iter_and_avg_result(r_type=self.__result_type,
                                                      files_per_iter=files_per_iter)
        row_ind = self.__add_row(exp_type_name=exp_type_name,
                                 model_dir=model_dir)

        # saving results.
        self.__save_results(it_results=it_results,
                            avg_res=self.__calc_avg_it_res(it_results),
                            folding_type=folding_type,
                            row_ind=row_ind,
                            labels_count=labels_count)

    def save(self, round_decimals):

        # Perform rounding.
        rounded_df = self.__df.round(decimals=round_decimals)

        # composing output filepath.
        basename = self._create_output_basename()
        filepath = basename + '.tex'
        print "Saving: {}".format(filepath)

        rounded_df.to_latex(filepath, na_rep='', index=False)

        print tabulate(rounded_df, headers='keys', tablefmt='psql')

    def register(self, model_name, folding_type, labels_count, ra_version):
        assert(isinstance(model_name, ModelNames))
        assert(isinstance(folding_type, FoldingType))
        assert(isinstance(ra_version, RuAttitudesVersions) or ra_version is None)

        folding_str = u"cv{}".format(self.__cv_count) \
            if folding_type == FoldingType.CrossValidation else u"fixed"
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

    def _create_model_dir(self, folding_type, model_name, exp_type_name):
        origin_name = super(FineTunedResultsProvider, self)._create_model_dir(folding_type=folding_type,
                                                                              model_name=model_name,
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


def fill_single23(res):
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


def fill_finetunned(res, labels):
    assert(isinstance(res, FineTunedResultsProvider))
    for model_name in ModelNames:
        for folding_type in FoldingType:
            for l in labels:
                res.register(model_name=model_name,
                             folding_type=folding_type,
                             labels_count=l,
                             ra_version=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tex table Gen")

    parser.add_argument('--results-dir',
                        dest='output_dir',
                        type=unicode,
                        nargs='?',
                        default=CustomNetworkExperimentIO.default_sources_dir,
                        help='Results dir')

    parser.add_argument('--result-type',
                        dest='result_type',
                        type=unicode,
                        nargs='?',
                        default=ResultType.F1.value,
                        choices=[rt.value for rt in ResultType],
                        help="Metric selection which will be used for table cell values")

    parser.add_argument('--training-type',
                        dest='training_type',
                        type=unicode,
                        nargs='?',
                        default=u'single',
                        choices=[u'single', u'ft'],
                        help='Training format used for results gathering')

    parser.add_argument('--labels',
                        dest='labels',
                        type=int,
                        nargs='*',
                        default=[2, 3],
                        help='Used labels')

    parser.add_argument('--foldings',
                        dest='foldings',
                        type=unicode,
                        nargs='*',
                        default=[f.value for f in FoldingType],
                        help="Used foldings")

    parser.add_argument('--round',
                        dest='round',
                        type=int,
                        nargs='?',
                        default=3,
                        help='Decimals rounding for float values')

    ModelInputTypeArg.add_argument(parser)

    args = parser.parse_args()

    # Reading arguments.
    training_type = args.training_type
    result_type = ResultType.from_str(args.result_type)
    model_input_type = ModelInputTypeArg.read_argument(args)
    provide_cv_columns = result_type != ResultType.TrainingEpochTime
    foldings = [FoldingType.from_str(v) for v in args.foldings]
    labels = args.labels
    cv_count = DEFAULT_CV_COUNT

    results_types = {
        u'single': ResultsTable,
        u'ft': FineTunedResultsProvider
    }

    class_type = results_types[training_type]

    # Single training format.
    rt = class_type(output_dir=args.output_dir,
                    input_type=model_input_type,
                    result_type=result_type,
                    labels=labels,
                    cv_count=cv_count,
                    foldings=foldings)

    # Filling
    if training_type == u'single':
        fill_single23(rt)
    if training_type == u'ft':
        fill_finetunned(rt, labels=labels)

    rt.save(args.round)
