import datetime
from dateutil import parser


EPOCH_ARGUMENT = u"Epoch"
AVG_FIT_COST_ARGUMENT = u"avg_fit_cost"
AVG_FIT_ACC_ARGUMENT = u"avg_fit_acc"


def get_message(epoch_index, avg_fit_cost, avg_fit_acc):
    """ Providing logging message
    """
    key_value_fmt = u"{k}: {v}"
    time = str(datetime.datetime.now())
    epochs = key_value_fmt.format(k=EPOCH_ARGUMENT, v=format(epoch_index))
    avg_fc = key_value_fmt.format(k=AVG_FIT_COST_ARGUMENT, v=avg_fit_cost)
    avg_ac = key_value_fmt.format(k=AVG_FIT_ACC_ARGUMENT, v=avg_fit_acc)
    return u"{time}: {epochs}: {avg_fc}, {avg_ac}".format(time=time,
                                                          epochs=epochs,
                                                          avg_fc=avg_fc,
                                                          avg_ac=avg_ac)


def extract_last_param_value_from_training_log(filepath, key):
    v = 0
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if key == AVG_FIT_ACC_ARGUMENT:
                v = float(line.split(u': ')[-1])
            elif key == EPOCH_ARGUMENT:
                # Extracting from line.
                epoch_index = int(line.split(u': ')[2])
                # We provide an amount, therefore we inc the latter by 1.
                v = epoch_index + 1

    return v


def chop_microseconds(delta):
    return delta - datetime.timedelta(microseconds=delta.microseconds)


def extract_avg_epoch_time_from_training_log(filepath):
    times = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            string_time = line.split(u': ')[0]
            times.append(parser.parse(string_time))

    deltas = []
    for e_index in range(1, len(times)-1):
        deltas.append(times[e_index+1] - times[e_index])

    if len(deltas) > 0:
        # NOTE. due to that the time may consider
        # computations of other operations therefore
        # we cannot gauaratee the correctness.
        # Hence we providing the minimal time.
        return chop_microseconds(sorted(deltas)[0])
    else:
        # Providing 0 by default otherwise.
        return datetime.timedelta()


def parse_float_network_parameter(filepath, param_name):
    # ['base:current_time', datetime.datetime(2020, 12, 30, 9, 30, 6, 531903)]
    # ['base:use_class_weights', True]
    # ['base:dropout (keep prob)', 0.8]
    # ['base:classes_count', 3]
    # ['base:class_weights', [33.333333333333336, 33.333333333333336, 33.333333333333336]]
    # ['base:terms_per_context', 50]

    value = 0
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if param_name not in line:
                continue

            # removing brackets
            line = line.strip()
            line = line[1:-1]
            # extracting value
            value = float(line.split(',')[-1])

    return value


