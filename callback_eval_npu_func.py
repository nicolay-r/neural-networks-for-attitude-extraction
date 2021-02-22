from arekit.common.evaluation.evaluators.three_class import ThreeClassEvaluator
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.experiment.data_type import DataType
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.common.utils import progress_bar_iter


def calc_f1_npu(doc_ids, synonyms, data_type,
                iter_etalon_opins_by_doc_id_func,
                iter_result_opins_by_doc_id_func):
    """ Provides f1 (neg, pos, neu) calculation by given enumerations of
        etalon and results opinions for a particular document (doc_id).
    """
    assert(isinstance(synonyms, SynonymsCollection))
    assert(isinstance(data_type, DataType))
    assert(callable(iter_etalon_opins_by_doc_id_func))
    assert(callable(iter_result_opins_by_doc_id_func))

    cmp_pairs_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
        doc_ids=doc_ids,
        read_etalon_collection_func=lambda doc_id: OpinionCollection(
            opinions=iter_etalon_opins_by_doc_id_func(doc_id),
            synonyms=synonyms,
            error_on_duplicates=False,
            error_on_synonym_end_missed=True),
        read_result_collection_func=lambda doc_id: OpinionCollection(
            opinions=iter_result_opins_by_doc_id_func(doc_id),
            synonyms=synonyms,
            error_on_duplicates=False,
            error_on_synonym_end_missed=False))

    # getting evaluator.
    evaluator = ThreeClassEvaluator(data_type)

    # evaluate every document.
    logged_cmp_pairs_it = progress_bar_iter(cmp_pairs_iter, desc=u"Evaluate", unit=u'pairs')
    return evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)