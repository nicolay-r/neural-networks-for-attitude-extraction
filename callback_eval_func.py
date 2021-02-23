from arekit.common.evaluation.evaluators.three_class import ThreeClassEvaluator
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.utils import progress_bar_iter


def calculate_results(doc_ids, evaluator,
                      iter_etalon_opins_by_doc_id_func,
                      iter_result_opins_by_doc_id_func):
    """ Provides f1 (neg, pos, neu) calculation by given enumerations of
        etalon and results opinions for a particular document (doc_id).
    """
    assert(isinstance(evaluator, ThreeClassEvaluator))
    assert(callable(iter_etalon_opins_by_doc_id_func))
    assert(callable(iter_result_opins_by_doc_id_func))

    cmp_pairs_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
        doc_ids=doc_ids,
        read_etalon_collection_func=lambda doc_id: iter_etalon_opins_by_doc_id_func(doc_id),
        read_result_collection_func=lambda doc_id: iter_result_opins_by_doc_id_func(doc_id))

    # evaluate every document.
    logged_cmp_pairs_it = progress_bar_iter(cmp_pairs_iter, desc=u"Evaluate", unit=u'pairs')
    return evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)


def create_etalon_with_neutral(collection, etalon_opins, neut_opins):
    assert(isinstance(collection, OpinionCollection))
    assert(len(collection) == 0)

    def __it_all():

        for etalon_opin in etalon_opins:
            yield etalon_opin

        for neut_opinion in neut_opins:
            yield neut_opinion

    for o in __it_all():
        if not collection.has_synonymous_opinion(o):
            collection.add_opinion(o)

    return collection
