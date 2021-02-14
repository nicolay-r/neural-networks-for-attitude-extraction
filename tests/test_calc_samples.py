import unittest

import sys

from samples_utils import calculate_samples_count

sys.path.append('../')


class TestCalcSamples(unittest.TestCase):

    __input_samples_filepath = u"data/test_sample_3l.tsv.gz"

    def test(self):
        print calculate_samples_count(input_samples_filepath=self.__input_samples_filepath,
                                      uint_label=0)

        print calculate_samples_count(input_samples_filepath=self.__input_samples_filepath,
                                      uint_label=1)

        print calculate_samples_count(input_samples_filepath=self.__input_samples_filepath,
                                      uint_label=2)


if __name__ == '__main__':
    unittest.main()
