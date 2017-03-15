""" Testing functions from test_functions.py
"""

import test_functions as tf

# test functions must be named "test_*"
def test_adder():
    assert tf.adder(2, 3) == 5

def test_multiply():
    assert tf.multiply(4,5) == 20


# test classes must be named "Test*"
class TestTwoFunctions(object):

    def test_absolute(self):
        assert tf.absolute(0) == 0
        assert tf.absolute(1) == 1
        assert tf.absolute(-4) == 4
        assert tf.absolute(-0.1) == 0.1

    def test_string_split(self):
        x = "hello"
        assert tf.string_split(x) == ["hello"]
        assert tf.string_split("hello world") == ["hello", "world"]
