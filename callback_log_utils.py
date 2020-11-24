from arekit.common.evaluation.results.two_class import TwoClassEvalResult


def create_verbose_eval_results_msg(eval_result, data_type, epoch_index):
    assert (isinstance(eval_result, TwoClassEvalResult))
    title = u"Stat for [{dtype}], e={epoch}:".format(dtype=data_type, epoch=epoch_index)
    contents = [u"{doc_id}: {result}".format(doc_id=doc_id, result=result)
                for doc_id, result in eval_result.iter_document_results()]
    return u'\n'.join([title] + contents)


def create_overall_eval_results_msg(eval_result, data_type, epoch_index):
    assert (isinstance(eval_result, TwoClassEvalResult))
    title = u"Stat for '[{dtype}]', e={epoch}".format(dtype=data_type, epoch=epoch_index)
    params = [u"{}: {}".format(metric_name, round(value, 2))
              for metric_name, value in eval_result.iter_results()]
    contents = u"; ".join(params)
    return u'\n'.join([title, contents])
