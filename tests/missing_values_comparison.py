"""Comparison the missing values replacement strategies."""

import time
import helpers
import missing_values as mv
import promethee as prom
import data_reader as dr
import copy
from scipy import stats
import numpy as np
import random


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


def compare_evaluations(alt_num=20, it=500, del_num=1):
    """Compare strategies."""
    random.seed(1)
    datasets = ('HR', 'SHA', 'EPI', 'HP')
    # datasets = ('SHA',)
    header = ["    "] + list(datasets) + ["mean", "std"]
    methods = {  # 'sreg': mv.replace_by_sreg,
               # 'creg': mv.replace_by_creg,
               # 'ereg': mv.replace_by_ereg,
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
            MSE = compare_evaluations_once(all_alts, alt_num, weights, del_num,
                                           methods)
            # print(taus)
            for method in methods:
                results_dataset[method].append(MSE[method])

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


def compare_evaluations_once(all_alts, alt_num, weights, del_number, methods):
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
