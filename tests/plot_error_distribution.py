"""Plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
import helpers
import data_reader as dr
# import pandas as pd
# from scipy import stats, integrate


def plot_dist(dataset):
    """Plot distributions.

    i,   c,  ev,   reg,  dom,  diff,  dk,  dk2,  knn,  mean,   med.
    """
    filename = 'res/error_distribution/' + dataset + '/'  \
               + dataset + '_errors.csv'
    filename = 'res/local_regression/' + dataset + '/'  \
               + 'errors.csv'
    header, errors = dr.open_errors(filename)

    # errors = errors[:25]

    errors_by_c = []
    c = 0
    n_sorted = 0
    while len(errors) > n_sorted:
        error_c = [error[3:] for error in errors if error[1] == c]
        n_sorted += len(error_c)
        print(len(error_c))
        errors_by_c.append(error_c)
        c += 1

    methods = header[3:]
    print(methods)
    # for crit in range(c):
    for crit in range(c):
        errors = errors_by_c[crit]
        fig = plt.figure(c, figsize=(16, 12), dpi=880)
        # fig = plt.figure(1)
        ax = fig.gca()
        # ax.set_autoscale_on(False)
        title = dataset + ' - Criterion ' + str(crit + 1)
        save_title = "res/local_regression/" + dataset + "/" + title + '.pdf'
        fig.canvas.set_window_title(title)
        for i, method in enumerate(methods):
            plt.subplot(320 + i + 1)
            error_method = [err[i] for err in errors]
            n, bins, patches = plt.hist(error_method, normed=1,
                                        facecolor='green', alpha=0.75)
            plt.title(method)
            # plt.axis([-1, 1, 0, 5])     # [xmin,xmax,ymin,ymax]
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=None, hspace=0.8)
        plt.savefig(save_title, bbox_inches='tight')
        # plt.show()
        plt.close()


def main():
    """Perform some tests on the plotting."""
    datasets = ('HDI', 'SHA', 'HP', 'CPU')
    for d in datasets:
        plot_dist(d)
    # t1 = np.arange(0.0, 5.0, 0.1)
    # t2 = np.arange(0.0, 5.0, 0.02)

    # plt.figure(1)
    # plt.subplot(223)
    # plt.plot(t1, f(t1), 'b')    # , t2, f(t2), 'k')

    # plt.subplot(212)
    # plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
    # plt.show()
