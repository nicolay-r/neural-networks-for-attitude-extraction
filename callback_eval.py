from os.path import join

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.bert.callback import Callback
from callback_log_iter import create_iteration_short_eval_msg, create_iteration_verbose_eval_msg
from common import Common


class CallbackEvalF1NPU(Callback):

    __log_eval_iter_verbose_filename = u"cb_eval_verbose_{iter}_{dtype}.log"

    def __init__(self, data_type):
        self.__data_type = data_type

        self.__eval_verbose_file = None
        self.__eval_short_file = None

        self.__it_index = None
        self.__log_dir = None

    def set_iter_index(self, it_index):
        self.__it_index = it_index

    def set_log_dir(self, target_dir):
        assert (isinstance(target_dir, unicode))
        self.__log_dir = join(target_dir, Common.log_dir)

    def write_results(self, result, data_type, epoch_index):
        eval_verbose_msg = create_iteration_verbose_eval_msg(eval_result=result,
                                                             data_type=data_type,
                                                             epoch_index=epoch_index)

        eval_short_msg = create_iteration_short_eval_msg(eval_result=result,
                                                         data_type=data_type,
                                                         epoch_index=epoch_index)

        # Saving results.
        self.__eval_verbose_file.write(u"{}\n".format(eval_verbose_msg))
        self.__eval_short_file.write(u"{}\n".format(eval_short_msg))

    def __enter__(self):
        # Compose filepath for verbose results.
        eval_verbose_log_filepath = join(
            self.__log_dir,
            self.__log_eval_iter_verbose_filename.format(iter=self.__it_index, dtype=self.__data_type))

        eval_short_log_filepath = join(
            self.__log_dir,
            Common.create_log_eval_filename(iter_index=self.__it_index, data_type=self.__data_type))

        create_dir_if_not_exists(eval_verbose_log_filepath)
        create_dir_if_not_exists(eval_short_log_filepath)

        print u"Saving at: {}".format(eval_short_log_filepath)

        self.__eval_short_file = open(eval_short_log_filepath, u"w", buffering=0)
        self.__eval_verbose_file = open(eval_verbose_log_filepath, u"w", buffering=0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__eval_verbose_file is not None:
            self.__eval_verbose_file.close()

        if self.__eval_short_file is not None:
            self.__eval_short_file.close()

        self.__it_index = None
