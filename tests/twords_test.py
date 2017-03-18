""" Testing methods in Twords

Note: although Twords tries to carefully store all strings as unicode, the
assert statements here consider a regular python string and a unicode string
as equal (Twords was written in python 2.7).
"""
import sys
import os
sys.path.append('../twords')

from twords.twords import Twords
import pytest

from numpy.testing import assert_approx_equal
import csv

class TestAttributeCreation(object):

    """def __init__(self):
        self.sample_background_1 = "sample_background_data.csv"
        self.sample_java_data = "sample_java_data.csv"
        """
    # sample_java_data = "sample_java_data.csv"

    def test_set_Search_terms_1(self):
        twit = Twords()
        twit.set_Search_terms(["term1"])
        assert twit.search_terms == ["term1"]

    def test_set_Search_terms_2(self):
        twit = Twords()
        twit.set_Search_terms([u"term1"])
        assert twit.search_terms == [u"term1"]

    def test_set_Search_terms_3(self):
        twit = Twords()
        with pytest.raises(AssertionError):
            twit.set_Search_terms("term1")

    def test_set_Search_terms_4(self):
        twit = Twords()
        with pytest.raises(AssertionError):
            twit.set_Search_terms(["term1", "term2", 2])


    def test_create_Background_dict_1(self):
        # name of file in tests directory
        sample_background_1 = "sample_background_data.csv"

        # joins absolute path of tests directory with "sample_background_data.csv"
        sample_background_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),sample_background_1)
        twit = Twords()
        twit.background_path = sample_background_1
        twit.create_Background_dict()

        # read in sample background data manually to compare
        def read_data(data):
            with open(data, 'r') as f:
                data = [row for row in csv.reader(f.read().splitlines())]
            return data

        background_data = read_data(sample_background_1)
        background_data = background_data[1:]
        background_dict = {unicode(line[0]): (float(line[2]), int(line[1]))
                           for line in background_data}

        for key in background_dict.keys():
            # compare frequency rate - a float
            assert_approx_equal(background_dict[key][0], twit.background_dict[key][0], 10)
            # compare occurrences - an integer
            assert background_dict[key][1] == twit.background_dict[key][1]

    def test_add_stop_words_1(self):
        # test adding single term
        # note:
        twit = Twords()
        twit.create_Stop_words()
        current_stop_words = twit.stop_words
        twit.add_stop_words("candycane")
        twit.add_stop_words(u'elephant')
        assert twit.stop_words == current_stop_words + [u"candycane", u'elephant']

    def test_add_stop_words_2(self):
        # test adding list of terms
        twit = Twords()
        twit.create_Stop_words()
        current_stop_words = twit.stop_words
        twit.add_stop_words(["marco", "polo"])
        assert twit.stop_words == current_stop_words + [u"marco", u"polo"]
