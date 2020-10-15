from callback import CustomCallback
from rusentrel.classic.common import classic_common_callback_modification_func
from rusentrel.rusentrel_ds.common import ds_common_callback_modification_func


def classic_cv_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    assert(isinstance(callback, CustomCallback))

    classic_common_callback_modification_func(callback)
    callback.set_key_save_hidden_parameters(False)

    callback.set_cancellation_acc_bound(0.981)
    callback.set_cancellation_f1_train_bound(0.85)
    callback.set_key_stop_training_by_cost(True)


def ds_cv_common_callback_modification_func(callback):
    """ This function describes configuration setup for all model callbacks.
        In case of training process that adopt distant-supervision approach.
    """
    assert(isinstance(callback, CustomCallback))

    ds_common_callback_modification_func(callback)
    callback.set_cancellation_acc_bound(0.999)
    callback.set_cancellation_f1_train_bound(0.85)
    callback.set_key_save_hidden_parameters(False)
    callback.set_key_stop_training_by_cost(True)