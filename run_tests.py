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
# Promethee_class.test_ranking('HDI')
# Promethee_class.test_ranking('SHA')
# Promethee_class.test_rr_counting_function()
# Promethee_class.test_rr_analysis(data[1])


# PrometheeMV_class.test_ranking('HDI')
# PrometheeMV_class.test_replacements()
# PrometheeMV_class.test_PMV()

missing_values_comparison.compare_rankings(alt_num=100, it=100, del_num=5)
# missing_values_comparison.test_guess_eval()
# missing_values_comparison.test_check_train_dom()

t2 = time.time()
print('test durations ::' + str(t2-t1))
