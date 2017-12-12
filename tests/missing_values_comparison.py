"""Comparison the missing values replacement strategies."""

import time
from helpers import NULL
import helpers
import regression as rg
import local_regression as lrg
import dominance_estimations as de
import knn
from sklearn.preprocessing import normalize
import missing_values as mv
import promethee as prom
import data_reader as dr
import copy
from scipy import stats
import numpy as np
import random
import csv


def compare_rankings_once(all_alts, alt_num, weights, del_number, methods):
    """Compare strategies once."""
    seed = random.randint(0, 1000)
    # print('seed', seed)
    # seed = 289
    # print(seed)
    alts = random.sample(all_alts, alt_num)

    alts_inc = mv.delete_l_evaluations(alts, del_number, seed)
    # print("gapped :")
    # helpers.printmatrix(alts_inc)

    PII = prom.PrometheeII(alts, weights=weights, seed=seed)
    ranking_PII = PII.ranking

    kendall_taus = {}
    for method in methods:
        alts_completed = methods[method](alts_inc)
        score = PII.compute_netflow(alts_completed)
        ranking = PII.compute_ranking(score)
        kendall_taus[method] = stats.kendalltau(ranking_PII, ranking)[0]

    return kendall_taus


def compare_rankings(alt_num=20, it=500, del_num=1):
    """Compare strategies."""
    random.seed(1)
    datasets = ('HR', 'SHA', 'EPI', 'HP')
    # datasets = ('SHA',)
    header = ["    "] + list(datasets) + ["mean", "std"]
    methods = {  # 'sreg': mv.replace_by_sreg,
               # 'creg': mv.replace_by_creg,
               # 'ereg': mv.replace_by_ereg,
               'sreg': mv.replace_by_sreg,
               'dom': mv.replace_by_dominance,
               'd_diff': mv.replace_by_dominance_smallest_diff,
               'knn': mv.replace_by_knn,
               'mean': mv.replace_by_mean,
               'med': mv.replace_by_med}
    #          'pij': mv.replace_by_pij}

    results = {method: [] for method in methods}
    meth_std = {method: [] for method in methods}

    for dataset in datasets:
        print('---------------------- ', dataset, ' -----------------------')
        t0 = time.time()
        results_dataset = {method: [] for method in methods}

        filename = 'data/' + dataset + '/raw.csv'
        all_alts, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]
        if weights == []:
            weights = None

        for i in range(it):
            taus = compare_rankings_once(all_alts, alt_num, weights, del_num,
                                         methods)
            # print(taus)
            for method in methods:
                results_dataset[method].append(taus[method])

        for method in methods:
            results[method].append(sum(results_dataset[method])/it)
            meth_std[method] += results_dataset[method]
        print('time:', time.time() - t0)

    final_matrix = [header]
    for m in methods:
        results[m].append(np.mean(results[m]))
        results[m].append(np.std(meth_std[m]))
        final_matrix.append([m] + results[m])

    helpers.printmatrix(final_matrix)


def compare_evaluations(alt_num=100, iterations=5,
                        outputdir='res/local_regression/'):
    """Compare strategies.

    Output in different files:
        1. All the errors for each dataset (prefix dataset):
            i, j, ev, reg, ...

        2. Statistics for each dataset (prefix dataset_statistics):
                 MEAN   STD
            reg
            ...

        3. Global statistics (prefix Global
                 SHA ... MEAN   STD
            reg
            ...
    """
    datasets = ('HDI', 'SHA', 'HP', 'CPU')
    # datasets = ('SHA',)
    global_header = ["    ", "mean", "std"]
    methods = {'reg': rg.get_regression,
               'lrg': lrg.get_estimation_by_local_regression,
               'dom': de.get_estimations_by_dominance,
               # 'diff': de.get_estimations_by_dominance_diff,
               # 'dk': de.get_estimations_by_dominance_knn,
               # 'dk2': de.get_estimations_by_dominance_knn_2,
               # 'dk3': de.get_estimations_by_dominance_knn_3,
               # 'dk4': de.get_estimations_by_dominance_knn_4,
               'knn': knn.get_knn,
               'mean': mv.get_mean,
               'med': mv.get_med}

    dataset_header = ['i', 'c', 'ev', 'lrg', 'reg', 'dom', 'diff', 'dk', 'dk2',
                      # 'dk3', 'dk4',
                      'knn', 'mean', 'med']

    row_methods_order = dataset_header[3:]

    global_res = {method: [] for method in methods}
    # global_std = {method: [] for method in methods}

    for dataset in datasets:
        print('---------------------- ', dataset, ' -----------------------')
        t0 = time.time()

        # output file for dataset
        dataset_output = outputdir + dataset + '.csv'
        dataset_statistics_output = outputdir + dataset + '_statistics.csv'

        dataset_res = []
        dataset_res.append(dataset_header)
        # used for std and mean
        dataset_res_dico = {method: [] for method in methods}

        filename = 'data/' + dataset + '/raw.csv'
        all_alts, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]

        A = random.sample(all_alts, alt_num)
        A = normalize(A, axis=0, copy=True, norm='max')
        A = [list(alt) for alt in A]

        for it in range(iterations):
            res_it = []
            i, c = random.randint(0, len(A)-1), random.randint(0, len(A[0])-1)

            res_it.append(i)
            res_it.append(c)

            ev = A[i][c]
            A[i][c] = NULL
            errors = compare_evaluations_once(A, ev, methods)
            A[i][c] = ev

            res_it.append(ev)

            for m in row_methods_order:
                res = errors[m]
                res_it.append(res)
                dataset_res_dico[m].append(res)

            dataset_res.append(res_it)

        helpers.matrix_to_csv(dataset_res, dataset_output)

        # Make the matrix for the statistics of the given dataset
        dataset_statistics_res = []
        dataset_statistics_res.append([dataset, "MEAN", "STD"])

        for method in methods:
            # keep all the errors for the global satistics
            global_res[method] += dataset_res_dico[method]

            line = [method, np.mean(dataset_res_dico[method]),
                    np.std(dataset_res_dico[method])]

            dataset_statistics_res.append(line)

        helpers.matrix_to_csv(dataset_statistics_res, dataset_statistics_output)

        print('time:', time.time() - t0)

    global_matrix = [global_header]
    for m in methods:
        std = np.std(global_res[m])
        mean = np.mean(global_res[m])
        global_matrix.append([m, mean, std])

    helpers.printmatrix(global_matrix)


def compare_evaluations_once(A_miss, ev, methods):
    """Compare strategies once."""
    errors = {}
    for method in methods:
        estimation = methods[method](A_miss)
        errors[method] = estimation

    for method in errors:
        if type(errors[method]) == str:
            print(method, errors[method])
            errors[method] = errors['mean']
            print(method, errors[method])

    for method in errors:
        # print(method, errors[method], ev)
        errors[method] = errors[method] - ev

    return errors


def test_check_train_dom(dataset="SHA", alt_num=100):
    """Check this function."""
    datasets = ('HR', 'SHA', 'EPI', 'HP')
    for dataset in datasets:
        print('---------------------- ', dataset, ' -----------------------')
        filename = 'data/' + dataset + '/raw.csv'
        all_alts, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]
        alts = random.sample(all_alts, alt_num)
        mv.check_train_dom(alts)


def test_guess_eval(dataset="SHA", alt_num=15, del_number=1, seed=0):
    """Test guess function."""
    filename = 'data/' + dataset + '/raw.csv'
    all_alts, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]
    alts = random.sample(all_alts, alt_num)

    alts = mv.delete_l_evaluations(alts, del_number, seed)
    mv.guess_all_bests_estimations(alts)
