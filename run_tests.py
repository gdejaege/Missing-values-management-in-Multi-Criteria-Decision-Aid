"""This file is used to launch the different test files.

Be aware that some tests could take hours (at least on my laptop).
All the parameters of the tests are optional. However, its should be easier for
you to define them.
"""
import time
from tests import *
import random

t1 = time.time()
data = ['HDI', 'SHA', 'EPI', 'GEQ']
random.seed(0)
random.seed()
# data_reader.test()
# Promethee_class.test_ranking('HDI')
# Promethee_class.test_ranking('SHA')
# Promethee_class.test_rr_counting_function()
# Promethee_class.test_rr_analysis(data[1])
# Promethee_class.test_ranking_SHA()


# PrometheeMV_class.test_ranking('HDI')
# PrometheeMV_class.test_replacements()
# PrometheeMV_class.test_PMV()

# missing_values_comparison.compare_rankings(alt_num=100, it=300, del_num=1)
# missing_values_comparison.compare_evaluations(alt_num=100, iterations=3500)
# missing_values_comparison.test_guess_eval()
# missing_values_comparison.test_check_train_dom()

# plot_error_distribution.main()

# dominance_estimation.check_dominance_assumption(iterations=500)
# dominance_estimation.check_if_dominance_interval(iterations=500)
# dominance_estimation.check_good_dominance_interval(iterations=500)

# make_fake_dataset.main('test/raw_noise', lambda x: 17 +  9*x - 9*x**2 + x**3,
#                        [i/10 for i in range(0, 101)], noise=True)
# make_fake_dataset.main('inverse', lambda x: 1/x,
#                        [0.25 + i*3.75/100 for i in range(0, 101)])

# check_intervals_knn.main()

# save_errors_to_file.main()
# missing_values_plot.main('HDI_20')
# missing_values_plot.main('fake/inverse')
# missing_values_plot.main('fake/test')
# missing_values_plot.main('fake/test_noise')
# layer_regression.main()

t2 = time.time()
print('test durations ::' + str(t2-t1))
