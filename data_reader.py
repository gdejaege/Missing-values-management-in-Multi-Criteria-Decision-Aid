"""Module used to read data-files or write results.

expected file hierarchy :
    /main : here lies our code files
        /data
            /Sample1
            /Sample2
            ...
        /res
            ...
"""
import csv
import string
import time
import os
import numpy as np


data_sets = ['data/HDI/', 'data/EPI/', 'data/GEQ/', 'data/SHA/']


def open_raw(filename):
    """Open raw csv-data-files.

    All data must be written is csv format.
    Different sections are separated with lines of '#'.
    The structure of the different sections is defined by the first line.
    """
    data = []
    with open(filename, newline='') as csvfile:
        content = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in content:
            data.append(row)

    header = data[0]
    weights = []
    coefficients = []
    alternatives = []
    thresholds = []
    titles = ['alternatives', 'weights', 'coefficients', 'thresholds']
    matrix = [alternatives, weights, coefficients, thresholds]
    i = 0
    ind = titles.index(header[i].lower().strip())
    for row in data[2:]:
        if row[0][0] == '#':
            i += 1
            ind = titles.index(header[i].lower().strip())
        else:
            matrix[ind].append(list(map(lambda x: float(x), row[:])))
    for i in range(1, len(matrix)):
        if matrix[i]:
            matrix[i] = matrix[i][0]
    return matrix


def open_errors(filename):
    """File with.

    i,   c,  ev,   eg,  dom,  diff,  dk,  dk2,  knn,  mean,   med.
    return this header + corresponding values (errors)
    """
    data = []
    with open(filename, newline='') as csvfile:
        content = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in content:
            if i != 0:
                clean_row = [float(el) for el in row if el.strip() != '']
            else:
                clean_row = row[:-1]
                i = 1
            data.append(clean_row)

    header = data[0]
    errors = data[1:]

    return header, errors


def write_raw(alternatives, filename):
    """write classic csv-data-files.

    write set of alternatives to a csv file.
    """
    output = open(filename, 'w')
    for alt in alternatives:
        for i in range(len(alt)):
            if i != len(alt)-1:
                print(str(alt[i]), end=',', file=output)
            else:
                print(str(alt[i]), file=output)
