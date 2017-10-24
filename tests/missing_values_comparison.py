"""Comparison the missing values replacement strategies."""

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
    random.seed(0)
    datasets = ('HR', 'SHA', 'EPI', 'HP')
    # datasets = ('SHA', 'EPI')
    header = ["    "] + list(datasets) + ["Total"]
    methods = {'sreg': mv.replace_by_sreg,
               'creg': mv.replace_by_creg,
               'ereg': mv.replace_by_ereg,
               'knn': mv.replace_by_knn,
               'mean': mv.replace_by_mean,
               'med': mv.replace_by_med}
    #          'pij': mv.replace_by_pij}

    results = {method: [] for method in methods}

    for dataset in datasets:
        print('---------------------- ', dataset, ' -----------------------')
        results_dataset = {method: 0 for method in methods}

        filename = 'data/' + dataset + '/raw.csv'
        all_alts, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]
        random.shuffle(all_alts)
        if weights == []:
            weights = None

        for i in range(it):
            taus = compare_rankings_once(all_alts, alt_num, weights, del_num,
                                         methods)
            # print(taus)
            for method in methods:
                results_dataset[method] += taus[method]

        for method in methods:
            results[method].append(results_dataset[method]/it)

    final_matrix = [header]
    for m in methods:
        results[m].append(np.mean(results[m]))
        final_matrix.append([m] + results[m])

    helpers.printmatrix(final_matrix)
