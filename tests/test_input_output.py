import unittest
import os
from desc.input_output import read_input

class TestIO(unittest.TestCase):
    """tests for input/output functions"""

    def test_min_input(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'MIN_INPUT')
        inputs = read_input(filename)
        
        self.assertEqual(len(inputs), 24)