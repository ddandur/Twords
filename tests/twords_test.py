""" Testing methods in Twords
"""

from twords import Twords
import pytest

class TestAttributeCreation(object):

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


    def test_set_Search_terms_4(self):
        twit = Twords()
        with pytest.raises(AssertionError):
            twit.set_Search_terms(["term1", "term2", 2])

            









    #def test_two(self):
    #    x = "hello"
    #    assert tf.string_split(x) == ["hello"]
    #    assert tf.string_split("hello world") == ["hello", "world"]
