import unittest
from os.path import dirname, join

from enum import Enum

from arekit.common.evaluation.evaluators.three_class import ThreeClassEvaluator
from arekit.common.evaluation.results.three_class import ThreeClassEvalResult
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.utils import progress_bar_iter
from arekit.contrib.experiments.rusentrel.labels_formatter import RuSentRelNeutralLabelsFormatter
from arekit.contrib.experiments.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.contrib.source.zip_utils import ZipArchiveUtils
from arekit.processing.lemmatization.mystem import MystemWrapper


class Results(Enum):
    Etalon = u'annot-3-scale'
    Test = u'test_results'


class CustomRuSentRelLabelsFormatter(RuSentRelLabelsFormatter):

    def __init__(self):
        super(CustomRuSentRelLabelsFormatter, self).__init__()

        neut_fmt = RuSentRelNeutralLabelsFormatter()
        for key, value in neut_fmt._stol.iteritems():
            self._stol[key] = value


class CustomZippedResultsIOUtils(ZipArchiveUtils):

    @staticmethod
    def get_archive_filepath(result_version):
        return join(dirname(__file__), u"./data/{version}.zip".format(version=result_version))

    @staticmethod
    def iter_doc_ids(result_version):
        for f_name in CustomZippedResultsIOUtils.iter_filenames_from_zip(result_version):
            doc_id_str = f_name.split('.')[0]
            yield int(doc_id_str)

    @staticmethod
    def iter_doc_opinions(doc_id, result_version, labels_fmt, opin_path_fmt=u"{doc_id}.opin.txt"):
        assert(isinstance(labels_fmt, StringLabelsFormatter))

        return CustomZippedResultsIOUtils.iter_from_zip(
            inner_path=join(opin_path_fmt.format(doc_id=doc_id)),
            process_func=lambda input_file: RuSentRelOpinionCollectionFormatter._iter_opinions_from_file(
                input_file=input_file,
                labels_formatter=labels_fmt),
            version=result_version)


class TestEvalF1NPU(unittest.TestCase):

    @staticmethod
    def iter_with_neutral(doc_id):

        # Providing original opinions from etalon data.
        for etalon_opinion in RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id):
            yield etalon_opinion

        # Providing manually annotated (neutral opinions)
        opins_it = CustomZippedResultsIOUtils.iter_doc_opinions(
            doc_id=doc_id,
            result_version=Results.Etalon,
            labels_fmt=CustomRuSentRelLabelsFormatter(),
            opin_path_fmt=u"art{doc_id}.neut.Test.txt")

        for neut_opinion in opins_it:
            yield neut_opinion

    def test(self):

        stemmer = MystemWrapper()
        actual_synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
            stemmer=stemmer,
            version=RuSentRelVersions.V11)

        doc_ids = [46]

        cmp_pairs_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=doc_ids,
            read_etalon_collection_func=lambda doc_id: OpinionCollection(
                opinions=self.iter_with_neutral(doc_id=doc_id),
                synonyms=actual_synonyms,
                error_on_duplicates=False,
                error_on_synonym_end_missed=True),
            read_result_collection_func=lambda doc_id: OpinionCollection(
                opinions=CustomZippedResultsIOUtils.iter_doc_opinions(
                    doc_id=doc_id,
                    labels_fmt=RuSentRelLabelsFormatter(),
                    result_version=Results.Test),
                synonyms=actual_synonyms,
                error_on_duplicates=False,
                error_on_synonym_end_missed=False))

        # getting evaluator.
        evaluator = ThreeClassEvaluator()

        # evaluate every document.
        logged_cmp_pairs_it = progress_bar_iter(cmp_pairs_iter, desc=u"Evaluate", unit=u'pairs')
        result = evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)
        assert (isinstance(result, ThreeClassEvalResult))

        # calculate results.
        result.calculate()

        # logging all the result information.
        for doc_id, doc_info in result.iter_document_results():
            print u"{}:\t{}".format(doc_id, doc_info)
        print "------------------------"
        print str(result.TotalResult)
        print "------------------------"


if __name__ == '__main__':
    unittest.main()
