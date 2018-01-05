"""Test the regression layer method efficiency."""

import time
from helpers import NULL
import helpers
import layer_regression as layer_rg
import random


def main(n=25):
    random.seed(0)
    # test_dominates()
    # test_layers()
    dataset = "SHA"
    A = helpers.get_dataset(dataset, n)
    k = len(A[0])
    i, j = random.randint(0, n-1), random.randint(0, k-1)
    print(A[i][j])
    A[i][j] = NULL

    helpers.printmatrix(A)

    res = layer_rg.layer_regression(A)

    print("\n", res)


def test_layers():
    random.seed(0)
    A = [[random.random() for i in range(3)] for j in range(15)]

    layers = layer_rg.compute_layers(A)

    for i, layer in enumerate(layers):
        print("layer :", i)
        helpers.printmatrix(layer)
        print()


def test_dominates():
    dominated = [4, 4, 3, 3]
    dominant = [5, 5, 5, 2]
    print(layer_rg.dominates(dominated, dominant))
