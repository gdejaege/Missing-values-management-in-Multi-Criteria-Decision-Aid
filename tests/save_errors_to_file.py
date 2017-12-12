"""Save the errors of different methods to files."""

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


def main(n=100, iterations=5000, outputdir='res/local_regression/'):
    """Save the errors of different methods to files."""
    datasets = ('HDI', 'SHA', 'HP', 'CPU')
    # datasets = ('SHA',)
    global_header = ["    ", "mean", "std"]
    methods = {'reg': rg.get_regression,
               'lrg': lrg.get_estimation_by_local_regression,
               'dom': de.get_estimations_by_dominance,
               'knn': knn.get_knn,
               'mean': mv.get_mean,
               'med': mv.get_med}

    sorted_meths = sorted([meth for meth in methods])
    header = ['i', 'c', 'ev'] + sorted_meths

    for dataset in datasets:
        print('---------------------- ', dataset, ' -----------------------')
        t0 = time.time()

        # output file
        output_file = outputdir + dataset + '/errors.csv'
        res = [header]
        A = helpers.get_dataset(dataset, n)
        k = len(A[0])

        for it in range(iterations):
            i, c = random.randint(0, n-1), random.randint(0, k - 1)

            ev = A[i][c]
            A[i][c] = NULL
            errors = compare_evaluations_once(A, ev, methods)
            A[i][c] = ev

            res.append([i, c, ev] + [errors[m] for m in sorted_meths])

        helpers.matrix_to_csv(res, output_file)
