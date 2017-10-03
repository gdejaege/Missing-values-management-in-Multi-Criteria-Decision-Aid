"""This file is used to launch the different test files.

Be aware that some tests could take hours (at least on my laptop).
All the parameters of the tests are optional. However, its should be easier for
you to define them.
"""
import time
from tests import *

t1 = time.time()
data = ['HDI', 'SHA', 'EPI', 'GEQ']

# data_reader.test()
# Promethee_class.test_ranking()
# Promethee_class.test_rr_counting_function()
# Promethee_class.test_rr_analysis(data[1])

missing_values_comparison.compare_rankings(data[1])

t2 = time.time()
print('test durations ::' + str(t2-t1))
