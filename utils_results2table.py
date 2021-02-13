import argparse
import collections

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
from arekit.contrib.networks.enum_name_types import ModelNames, ModelNamesService
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions, RuAttitudesVersionsService
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from callback_log_exp import parse_last_epoch_results
from callback_log_iter import parse_last
from callback_log_training import extract_avg_epoch_time_from_training_log, extract_last_param_value_from_training_log, \
    chop_microseconds, parse_float_network_parameter, AVG_FIT_ACC_ARGUMENT, EPOCH_ARGUMENT
from common import Common
from experiment_io import CustomNetworkExperimentIO


# Dictionary which allows us to find an attentive model
# According to the one related but without attentive mechanism.
# There are even more models, so this list could be enlarged
# and finally then moved into AREkit framework.
attentive_models_matching = {
    ModelNames.CNN: [ModelNames.AttEndsCNN],
    ModelNames.PCNN: [ModelNames.AttEndsPCNN],
    ModelNames.LSTM: [ModelNames.IANEnds],
    ModelNames.BiLSTM: [ModelNames.AttSelfPZhouBiLSTM]
}


class ResultType(Enum):

    F1 = u'f1'
    TrainingEpochTime = u'train-epoch-time'
    TrainingTotalTime = u'train-total-time'
    TrainingAccuracy = u'train-acc'
    F1Train = u'f1-last-train'
    EpochsCount = u'epochs'
    # Using WIMS-2020 related paper format for results improvement calucalation.
    # Considering f1-test values by default.
    DSDiffF1Improvement = u'ds-diff-imp'
    DSDiffAttImprovement = u'ds-diff-att'

    LearningRate = u'train-lr'

    @staticmethod
    def from_str(value):
        for t in ResultType:
            if t.value == value:
                return t


class ResultsEvalContext(object):

    def __init__(self, model_name, folding_type, ra_ver, rsr_ver, labels_count, rt):
        assert(isinstance(model_name, ModelNames))
        assert(isinstance(folding_type, FoldingType))
        assert(isinstance(ra_ver, RuAttitudesVersions) or ra_ver is None)
        assert(isinstance(rsr_ver, RuSentRelVersions))
        assert(isinstance(labels_count, int))
        assert(isinstance(rt, ResultType))
        self.model_name = model_name
        self.folding_type = folding_type
        self.ra_ver = ra_ver
        self.rsr_ver = rsr_ver
        self.labels_count = labels_count
        self.rt = rt

    @classmethod
    def copy(cls, other):
        assert(isinstance(other, ResultsEvalContext))
        return cls(model_name=other.model_name,
                   folding_type=other.folding_type,
                   ra_ver=other.ra_ver,
                   rsr_ver=other.rsr_ver,
                   labels_count=other.labels_count,
                   rt=other.rt)


class ResultsTable(object):

    MODEL_NAME_COL = u'model_name'
    DS_TYPE_COL = u'ds_type'
    
    sl_template = u"rsr-{rsr_version}-{folding_str}-balanced-tpc50_{labels_count}l"
    sl_ds_template = u"rsr-{rsr_version}-ra-{ra_version}-{folding_str}-balanced-tpc50_{labels_count}l"

    def __init__(self, input_type, output_dir, result_types, labels, cv_count, foldings):
        assert(isinstance(input_type, ModelInputType))
        assert(isinstance(output_dir, unicode))
        assert(isinstance(result_types, list))
        assert(isinstance(cv_count, int))
        assert(isinstance(labels, list))
        assert(isinstance(foldings, list))

        def __it_columns(rt, label):
            # Providing extra columns for results.
            if provide_cv_columns:
                # cv-based results columns.
                yield (self.__rcol_agg(rt, label, FoldingType.CrossValidation), float)
                for cv_index in range(cv_count):
                    yield (self.__rcol_it(rt, label, cv_index), float)
            if provide_fx_columns:
                # fx-based result columns.
                yield (self.__rcol_agg(rt, label, FoldingType.Fixed), float)

        provide_cv_columns = FoldingType.CrossValidation in foldings
        provide_fx_columns = FoldingType.Fixed in foldings

        # Composing table for results using dataframe.
        dtypes_list = [(self.MODEL_NAME_COL, unicode),
                       (self.DS_TYPE_COL, unicode)]

        # Providing result columns.
        for label in labels:
            for rt in result_types:
                assert(isinstance(rt, ResultType))
                for col in __it_columns(rt=rt, label=label):
                    dtypes_list.append(col)

        np_data = np.empty(0, dtype=np.dtype(dtypes_list))

        self.__df = pd.DataFrame(np_data)
        self.__input_type = input_type
        self.__output_dir = output_dir
        self.__result_types = result_types
        self.__cv_count = cv_count

    @staticmethod
    def __rcol_agg(rt, labels_count, folding_type):
        assert(isinstance(rt, ResultType))
        assert(isinstance(folding_type, FoldingType))
        return u"{rt}_{labels_count}_{folding}".format(
            rt=rt.value,
            labels_count=labels_count,
            folding=u'avg' if folding_type == FoldingType.CrossValidation else u'test')

    @staticmethod
    def __rcol_it(rt, labels_count, it):
        assert(isinstance(rt, ResultType))
        return u"{rt}_{labels_count}_cv{it}".format(rt=rt.value,
                                                    labels_count=labels_count,
                                                    it=it)

    def _create_output_basename(self):
        return u"results-{input_type}-{result_type}".format(
            input_type=self.__input_type.value,
            result_type=u"_".join([rt.value for rt in self.__result_types]))

    def _create_model_dir(self, folding_type, model_name, exp_type):
        return Common.create_full_model_name(folding_type=folding_type,
                                             input_type=self.__input_type,
                                             model_name=model_name)

    def __save_results(self, rt, it_results, avg_res,
                       labels_count, folding_type, row_ind):
        assert(isinstance(rt, ResultType))
        assert(isinstance(it_results, collections.Iterable))

        # set avg. result.
        col_name_avg = self.__rcol_agg(rt=rt, labels_count=labels_count, folding_type=folding_type)
        if col_name_avg in self.__df.columns:
            self.__df.set_value(row_ind, col_name_avg, avg_res)

        # setting up it_results
        if folding_type != FoldingType.CrossValidation:
            return

        # setup cv values.
        for cv_index, cv_value in enumerate(it_results):
            col_name = self.__rcol_it(rt=rt, labels_count=labels_count, it=cv_index)
            if col_name not in self.__df.columns:
                continue
            self.__df.set_value(row_ind, col_name, cv_value)

    def __iter_files_per_iteration(self, result_type, folding_type):
        assert(isinstance(folding_type, FoldingType))

        iters = self.__cv_count if folding_type == FoldingType.CrossValidation else 1

        if result_type == ResultType.F1 or \
            result_type == ResultType.DSDiffF1Improvement or \
            result_type == ResultType.DSDiffAttImprovement:
            yield join(Common.log_dir, Common.log_test_eval_exp_filename)
        elif result_type == ResultType.TrainingEpochTime or \
                result_type == ResultType.TrainingTotalTime or \
                result_type == ResultType.TrainingAccuracy or \
                result_type == ResultType.EpochsCount:
            for it_index in range(iters):
                yield join(Common.log_dir, Common.create_log_train_filename(data_type=DataType.Train,
                                                                            iter_index=it_index))
        elif result_type == ResultType.F1Train:
            for it_index in range(iters):
                yield join(Common.log_dir, Common.create_log_eval_filename(data_type=DataType.Train,
                                                                           iter_index=it_index))
        elif result_type == ResultType.LearningRate:
            for it_index in range(iters):
                yield join(Common.log_dir, Common.model_config_name)
        else:
            raise NotImplementedError("Not supported type: {}".format(result_type))

    def __parse_iter_results(self, files_per_iter, eval_ctx):
        assert(isinstance(files_per_iter, list))
        assert(isinstance(eval_ctx, ResultsEvalContext))

        # Picking a current result type for eval_context.
        r_type = eval_ctx.rt

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
            # Calculate epochs count.
            eval_ctx.rt = ResultType.EpochsCount
            epochs = self.__parse_iter_results(files_per_iter=files_per_iter,
                                               eval_ctx=eval_ctx)
            # Calculate training epoch time.
            eval_ctx.rt = ResultType.TrainingEpochTime
            times = self.__parse_iter_results(files_per_iter=files_per_iter,
                                              eval_ctx=eval_ctx)
            return [epochs[i] * times[i] for i in range(len(epochs))]
        elif r_type == ResultType.F1Train:
            return [parse_last(filepath=fp, col=TwoClassEvalResult.C_F1) for fp in files_per_iter]
        elif r_type == ResultType.LearningRate:
            return [parse_float_network_parameter(fp, u'learning_rate') for fp in files_per_iter]
        elif r_type == ResultType.DSDiffAttImprovement:
            # Perform calculation of attentive models over non-attentive.

            def __calc_diff(att_it_results, rt):
                assert(isinstance(rt, ResultType))

                # Calculate current results.
                local_eval_ctx = ResultsEvalContext.copy(eval_ctx)
                local_eval_ctx.rt = rt
                curr_it_results = self.__parse_iter_results(files_per_iter=files_per_iter,
                                                            eval_ctx=local_eval_ctx)

                diff = [self.__calc_diff_metric(a=att_it_results[i], b=curr_it_results[i])
                        for i in range(len(curr_it_results))]
                res.append(diff)

            # using this as a local variable which is accessible from callback
            res = []

            # Perform another experiments result evaluation.
            for att_model in attentive_models_matching[eval_ctx.model_name]:
                self._for_experiment(model_name=att_model,
                                     folding_type=eval_ctx.folding_type,
                                     ra_version=eval_ctx.ra_ver,
                                     rsr_version=eval_ctx.rsr_ver,
                                     labels_count=eval_ctx.labels_count,
                                     result_types=[ResultType.F1],
                                     callback=__calc_diff)

            # Calculating average diff
            return np.mean(res, axis=0) if len(res) > 0 else []

        elif r_type == ResultType.DSDiffF1Improvement:

            def __calc_diff(base_it_results, rt):
                assert(isinstance(rt, ResultType))

                # Calculate current results.
                local_eval_ctx = ResultsEvalContext.copy(eval_ctx)
                local_eval_ctx.rt = rt
                curr_it_results = self.__parse_iter_results(files_per_iter=files_per_iter,
                                                            eval_ctx=local_eval_ctx)

                # calculating result difference.
                diff = [self.__calc_diff_metric(a=curr_it_results[i], b=base_it_results[i])
                        for i in range(len(curr_it_results))]

                res.append(diff)

            # using this as a local variable which is accessible from callback
            res = []

            # Perform another experiments result evaluation.
            self._for_experiment(model_name=eval_ctx.model_name,
                                 folding_type=eval_ctx.folding_type,
                                 ra_version=None,
                                 rsr_version=eval_ctx.rsr_ver,
                                 labels_count=eval_ctx.labels_count,
                                 result_types=[ResultType.F1],
                                 callback=__calc_diff)

            return res[0] if len(res) == 1 else []
        else:
            raise NotImplementedError("Not supported type: {}". format(r_type))

    @staticmethod
    def __calc_diff_metric(a, b):
        """ Perform difference calculation in percents.
        """
        return(float(a) / b - 1) * 100 if abs(b-0) > 1e-5 else 0

    @staticmethod
    def __calc_avg_it_res(result_type, it_results):
        assert(isinstance(result_type, ResultType))

        if len(it_results) == 0:
            return 0

        if len(it_results) == 1:
            return it_results[0]

        if result_type == ResultType.TrainingEpochTime or \
            result_type == ResultType.TrainingTotalTime:
            # These complex implementation due to datetime results
            # the latter won't work with np.mean.
            total_result = it_results[0]
            for ind in range(1, len(it_results)):
                total_result += it_results[ind]
            return chop_microseconds(total_result / len(it_results))
        else:
            return np.mean(it_results)

    @staticmethod
    def __create_exp_type_or_none(ra_version):
        assert(isinstance(ra_version, RuAttitudesVersions) or ra_version is None)
        ra_type = None if ra_version is None else ra_version
        return ra_type.value if isinstance(ra_type, RuAttitudesVersions) else None

    def __add_or_find_existed_row(self, ra_version, folding_type, model_name):

        exp_type = self.__create_exp_type_or_none(ra_version)
        model_dir = self._create_model_dir(folding_type=folding_type,
                                           model_name=model_name,
                                           exp_type=exp_type)

        exp_type_name = exp_type if exp_type is not None else u'-'

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

    def _create_exp_dir(self, cv_count, ra_version, folding_type, labels_count, rsr_version):
        assert(isinstance(cv_count, int))
        assert(isinstance(ra_version, RuAttitudesVersions) or ra_version is None)
        assert(isinstance(rsr_version, RuSentRelVersions))

        folding_str = u"cv{}".format(cv_count) \
            if folding_type == FoldingType.CrossValidation else u"fixed"

        if ra_version is None:
            # Using supervised learning only
            exp_dir = ResultsTable.sl_template.format(rsr_version=rsr_version.value,
                                                      folding_str=folding_str,
                                                      labels_count=str(labels_count))
        else:
            # Using distant supervision in combination with supervised learning.
            exp_dir = ResultsTable.sl_ds_template.format(rsr_version=rsr_version.value,
                                                         ra_version=ra_version.value,
                                                         folding_str=folding_str,
                                                         labels_count=str(labels_count))

        return exp_dir

    def __for_result_type(self, eval_ctx, target_to_path):
        assert(isinstance(eval_ctx, ResultsEvalContext))
        assert(callable(target_to_path))

        folding_type = eval_ctx.folding_type

        # Composing the related files
        files_per_iter = [target_to_path(target) for
                          target in self.__iter_files_per_iteration(result_type=eval_ctx.rt,
                                                                    folding_type=folding_type)]

        # Check files existance.
        for target_file in files_per_iter:
            if not exists(target_file):
                return None

        return self.__parse_iter_results(files_per_iter=files_per_iter,
                                         eval_ctx=eval_ctx)

    def _for_experiment(self, model_name, folding_type, ra_version,
                        rsr_version, labels_count, result_types, callback):
        assert(isinstance(model_name, ModelNames))
        assert(isinstance(folding_type, FoldingType))
        assert(isinstance(result_types, list))
        assert(isinstance(labels_count, int))

        def __target_to_path(target):
            # This is how we combine all the parameters into final
            # path to the target.
            return join(self.__output_dir, exp_dir, model_dir, target)

        exp_type = self.__create_exp_type_or_none(ra_version)
        model_dir = self._create_model_dir(folding_type=folding_type,
                                           model_name=model_name,
                                           exp_type=exp_type)

        # Composing the result dir.
        exp_dir = self._create_exp_dir(ra_version=ra_version,
                                       folding_type=folding_type,
                                       labels_count=labels_count,
                                       rsr_version=rsr_version,
                                       cv_count=self.__cv_count)

        for rt in result_types:
            assert(isinstance(rt, ResultType))

            # Composing eval context that allows us
            # to additionally run results evaluation,
            # if the latter is needed.
            eval_ctx = ResultsEvalContext(ra_ver=ra_version,
                                          rsr_ver=rsr_version,
                                          model_name=model_name,
                                          labels_count=labels_count,
                                          rt=rt,
                                          folding_type=folding_type)

            # Calculate and filling results.
            it_results = self.__for_result_type(target_to_path=__target_to_path,
                                                eval_ctx=eval_ctx)

            if it_results is None:
                continue

            # otherwise we can cast it as follows
            callback(it_results, rt)

    def save(self, round_decimals, coef_scaler):
        assert(isinstance(coef_scaler, float))

        self.__df.replace('nan', np.nan, inplace=True)

        # Perform rounding.
        rounded_df = self.__df.round(decimals=round_decimals)
 
        # Perform results scaling (Optional).
        if (coef_scaler > 1.0):
            rounded_df[rounded_df.select_dtypes(include=['number']).columns] *= coef_scaler 

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

        def __save_results_into_table(it_results, rt):
            assert(isinstance(rt, ResultType))

            # Finding the related row_id.
            row_ind = self.__add_or_find_existed_row(ra_version=ra_version,
                                                     folding_type=folding_type,
                                                     model_name=model_name)

            # Calculate averaged result that needed
            # in for the related table column.
            avg_res = self.__calc_avg_it_res(it_results=it_results,
                                             result_type=rt)

            # Writing results into the related row.
            self.__save_results(it_results=it_results,
                                avg_res=avg_res,
                                folding_type=folding_type,
                                row_ind=row_ind,
                                rt=rt,
                                labels_count=labels_count)

        self._for_experiment(model_name=model_name,
                             folding_type=folding_type,
                             ra_version=ra_version,
                             labels_count=labels_count,
                             rsr_version=RuSentRelVersions.V11,
                             result_types=self.__result_types,
                             # Processing results by saving the latter into table.
                             callback=__save_results_into_table)


class FineTunedResultsProvider(ResultsTable):

    # Assuming that we using non-ra experiment
    # for fine-tunning.
    __source_ra_version = None

    __fine_tuned_suffix = u"{model_name}-ft-{model_tag}"

    __tags = {
        __source_ra_version: None,
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

    def _create_exp_dir(self, cv_count, ra_version, folding_type, labels_count, rsr_version):
        return super(FineTunedResultsProvider, self)._create_exp_dir(cv_count=cv_count,
                                                                     ra_version=self.__source_ra_version,
                                                                     folding_type=folding_type,
                                                                     labels_count=labels_count,
                                                                     rsr_version=rsr_version)

    def _create_model_dir(self, folding_type, model_name, exp_type):
        assert(isinstance(exp_type, unicode) or exp_type is None)

        origin_name = super(FineTunedResultsProvider, self)._create_model_dir(folding_type=folding_type,
                                                                              model_name=model_name,
                                                                              exp_type=exp_type)

        # In such case we would like to provide
        # original name, i.e. the case when we
        # do not adopt fine-tunning in training process
        if exp_type is None:
            return origin_name

        ra_version = RuAttitudesVersionsService.find_by_name(exp_type)

        return self.__fine_tuned_suffix.format(model_name=origin_name,
                                               model_tag=self.__model_tag_from_ra_version(ra_version))

    def register(self, model_name, folding_type, labels_count, ra_version):
        assert(ra_version is self.__source_ra_version)

        # For every tag key we gathering results
        # within a single experiment dir.
        for ra_version_loc in self.__tags.iterkeys():
            super(FineTunedResultsProvider, self).register(
                model_name=model_name,
                folding_type=folding_type,
                labels_count=labels_count,
                ra_version=ra_version_loc)


class PreTrainedResultsProvider(ResultsTable):

    pt_template = u"ra-{ra_version}-balanced-tpc50_{labels_count}l"

    def _create_exp_dir(self, cv_count, ra_version, folding_type, labels_count, rsr_version):
        return self.pt_template.format(ra_version=ra_version.value,
                                       labels_count=str(labels_count))


def fill_single23(res, models, ra_3l, ra_2l):
    assert(isinstance(res, ResultsTable))
    assert(isinstance(models, list))
    assert(isinstance(ra_2l, list))
    assert(isinstance(ra_3l, list))

    for model_name in models:
        assert(isinstance(model_name, ModelNames))
        for folding_type in FoldingType:
            # for 3-scale
            for ra_version in ra_3l:
                res.register(model_name=model_name,
                             folding_type=folding_type,
                             labels_count=3,
                             ra_version=ra_version)

            # for 2-scale
            for ra_version in ra_2l:
                res.register(model_name=model_name,
                             folding_type=folding_type,
                             labels_count=2,
                             ra_version=ra_version)


def fill_finetunned(res, models, labels):
    assert(isinstance(res, FineTunedResultsProvider))
    assert(isinstance(models, list))

    for model_name in models:
        assert(isinstance(model_name, ModelNames))
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
                        dest='result_types',
                        type=unicode,
                        nargs='*',
                        default=[ResultType.F1.value],
                        choices=[rt.value for rt in ResultType],
                        help="Metric selection which will be used for table cell values")

    parser.add_argument('--training-type',
                        dest='training_type',
                        type=unicode,
                        nargs='?',
                        default=u'single',
                        choices=[u'single', u'ft', u'pt'],
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

    parser.add_argument('--scale',
                        dest='scale',
                        type=float,
                        nargs='?',
                        default=1.0,
                        help='Scaler coefficient for all the results in table')

    parser.add_argument('--models',
                        dest='models',
                        type=unicode,
                        nargs='*',
                        # By default, we limited by those models that were
                        # mentioned earlier in Rusnachenko et. al. papers.
                        default=[m.value for m in Common.default_results_considered_model_names_list()],
                        choices=[m_name for m_name in ModelNamesService.iter_supported_names()],
                        help='List of model names')

    ModelInputTypeArg.add_argument(parser)

    args = parser.parse_args()

    # Reading arguments.
    training_type = args.training_type
    result_types = [ResultType.from_str(r) for r in args.result_types]
    model_input_type = ModelInputTypeArg.read_argument(args)
    foldings = [FoldingType.from_str(v) for v in args.foldings]
    labels = args.labels
    cv_count = DEFAULT_CV_COUNT

    # model names
    if ResultType.DSDiffAttImprovement in result_types:
        # We consider only those models that are
        # matched with the corresponding attentive models.
        models = [ModelNamesService.get_type_by_name(m_name) for m_name in args.models
                  if ModelNamesService.get_type_by_name(m_name) in attentive_models_matching]
    else:
        models = [ModelNamesService.get_type_by_name(m_name) for m_name in args.models]

    # Supported training types.
    training_types = {
        u'single': ResultsTable,
        u'ft': FineTunedResultsProvider,
        u'pt': PreTrainedResultsProvider
    }

    class_type = training_types[training_type]

    # Single training format.
    rt = class_type(output_dir=args.output_dir,
                    input_type=model_input_type,
                    result_types=result_types,
                    labels=labels,
                    cv_count=cv_count,
                    foldings=foldings)

    ra_3l = [RuAttitudesVersions.V20LargeNeut,
             RuAttitudesVersions.V20BaseNeut,
             RuAttitudesVersions.V12]

    ra_2l = [RuAttitudesVersions.V20Large,
             RuAttitudesVersions.V20Base,
             RuAttitudesVersions.V12]

    # Filling
    if training_type == u'single':
        fill_single23(rt, models=models, ra_3l=ra_3l + [None], ra_2l=ra_2l + [None])
    if training_type == u'ft':
        fill_finetunned(rt, labels=labels, models=models)
    if training_type == u'pt':
        fill_single23(rt, models=models, ra_3l=ra_3l, ra_2l=ra_2l)

    rt.save(round_decimals=args.round,
            coef_scaler=args.scale)
