""" functions to practice using pytest
"""
def adder(x_num, y_num):
    """ add two numbers together
    """
    return x_num + y_num

def multiply(x_num, y_num):
    """ return product of two numbers
    """
    return x_num*y_num


def absolute(x):
    """ Return absolute value of x
    """
    if x >= 0:
        return x
    else:
        return -x

def string_split(string):
    """ Return a list of substrings of string, where substrings are
        parts that are separated by whitespace
    """
    return string.split()
