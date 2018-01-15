"""Regression functions implementation."""

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from helpers import NULL
import helpers
from scipy import stats
from sklearn.metrics import mean_squared_error as sk_mean_squared_error


def replace_by_creg(alts):
    """Try to guess the missing[s] value[s] using correlated regressions."""
    return replace_by_reg(alts, 'correlation')


def replace_by_sreg(alts):
    """Try to guess the missing[s] value[s] using simple regressions."""
    return replace_by_reg(alts, 'simple')


def replace_by_ereg(alts):
    """Try to guess the missing[s] value[s] using error regressions."""
    return replace_by_reg(alts, 'error')


def replace_by_reg(alts, method):
    """Try to guess the missing[s] value[s] using the precised regression."""
    # random.shuffle(alts)
    # helpers.printmatrix(alts)
    incompletes = [alt for alt in alts if NULL in alt]
    complete_alts = [alt for alt in alts if NULL not in alt]

    completed_alts = copy.deepcopy(alts)
    for incomplete in incompletes:
        i = alts.index(incomplete)
        criteria = [k for k, x in enumerate(incomplete) if x == NULL]
        for c in criteria:
            evaluation = estimate_by_regression(complete_alts, c, incomplete,
                                                method)
            completed_alts[i][c] = evaluation
    return completed_alts


def get_regression(A):
    """Get the estimate of the regression."""
    incomplete = [alt for alt in A if NULL in alt][0]
    complete_alts = [alt for alt in A if NULL not in alt]
    c = incomplete.index(NULL)

    evaluation = estimate_by_regression(complete_alts, c, incomplete)
    return evaluation


def estimate_by_regression(A, c, a_miss):
    """Try to find a model to guess the evaluations on the given criterion."""
    k = len(A[0])
    A_c = [a[c] for a in A]
    A_but_c = [[a[i] for i in range(k) if i != c] for a in A]
    a_miss_but_c = [a_miss[i] for i in range(k) if i != c]
    a_miss_but_c = np.array(a_miss_but_c)
    a_miss_but_c = a_miss_but_c.reshape(1, -1)

    lm = linear_model.LinearRegression()

    lm.fit(A_but_c, A_c)
    estimation = lm.predict(a_miss_but_c)
    return estimation


def estimate_by_regression_old(alternatives, criterion, incomplete,
                               method='simple'):
    """Try to find a model to guess the evaluations on the given criterion."""
    # matrix of criteria instead of alternatives:
    criteria = list(map(list, zip(*alternatives)))

    # Begin with the machine learning notation : goal = y, data = x
    y = criteria[criterion]
    x_tr_transposed = criteria[:criterion] + criteria[criterion+1:]
    # training set in a n*k form
    x_tr = list(map(list, zip(*x_tr_transposed)))
    x_test = [ev for ev in incomplete if ev != NULL]

    correlations = [stats.pearsonr(y, xi)[0] for xi in x_tr_transposed]

    models = []
    for c in range(len(x_tr[0])):
        x_tr_c = [[a[c]] for a in x_tr]
        models.append(train_regression(y, x_tr_c))

    estimations = [float(model[0].predict(xi))
                   for model, xi in zip(models, x_test)]
    MSEs = [i[1] for i in models]
    MSEs = [i/sum(MSEs) for i in MSEs]

    best_ind = MSEs.index(min(MSEs))
    estimation = float(models[best_ind][0].predict(x_test[best_ind]))

    """
    if method == 'simple':
        best_ind = MSEs.index(min(MSEs))
        estimation = float(models[best_ind][0].predict(x_test[best_ind]))
        # print('Sreg replacement:', estimation)
    elif method == 'correlation':
        estimation = sum([e*r for e, r in zip(estimations, correlations)])
        estimation /= sum(correlations)
        # print('Correlation replacement:', estimation)
    elif method == 'error':
        estimation = sum([e*(1 - err) for e, err in zip(estimations, MSEs)])
        estimation /= sum([1 - i for i in MSEs])
        # print('Error replacement:', estimation)
    """

    return estimation


def train_regression(y, x,  folds=4):
    """Find the best simple regression evaluation."""
    # these lm need to fit an 1D array of the style [[a0], [a1], ...]
    n = len(y)
    part = n // folds
    lms = []
    MSEs = []
    # print('x tr : \n', x)
    # print('y : \n', y)
    # print()

    # for fold in range(1):
    for fold in range(folds):
        lm, MSE = regression(y, x, fold, part)
        MSEs.append(MSE)
        lms.append(lm)

    lms = sorted(lms, key=lambda model: MSEs[lms.index(model)])
    MSEs.sort()
    # print(MSEs)
    return lms[0], MSEs[0]


def regression(y, x, fold, part):
    """Perform and test a linear regression."""
    n = len(y)
    i_test = [j for j in range(fold*part, (fold+1)*part)]
    i_tr = [j for j in range(n) if j not in i_test]

    if type(x[0]) == float:
        x = [x]

    x_tr = [x[i] for i in i_tr]
    # x_tr2 = [x[i] for i in i_tr]
    x_test = [x[i] for i in i_test]
    # x_test2 = [x[i] for i in i_test]

    y_tr = [[y[i]] for i in i_tr]
    y_test = [y[i] for i in i_test]

    # helpers.print_transpose([x_tr2, y_tr2])

    lm = linear_model.LinearRegression()
    lm.fit(x_tr, y_tr)
    y_pred = [pred[0] for pred in lm.predict(x_test)]
    # helpers.print_transpose([x_test2, y_test2, y_pred])
    MSE = sk_mean_squared_error(y_pred, y_test)
    # print(mean_squared_error(y_pred, y_test))
    return lm, MSE
