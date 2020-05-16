import numpy
import math


def get_weight(tf):
    if tf and tf > 0:
        return 1+math.log10(tf)
    else:
        return 0


def wtf(matx):
    nmtx = []
    for row in matx:
        vals = []
        for value in row:
            vals.append(round(get_weight(value), 2))
        nmtx.append(vals)
    return nmtx


def normalize(mtx):
    n_cols = mtx.shape[1]
    new_mtx = []
    for i in range(n_cols):
        col = mtx[:, i]
        module = 0
        for value in col:
            module += value**2

        module = round(math.sqrt(module), 2)
        new_col = []
        for value in col:
            new_col.append(round(value/module, 2))
        new_mtx.append(new_col)

    new_mtx = numpy.column_stack(new_mtx)
    return new_mtx


def get_distance_matrix(mtx):
    new_mtx = numpy.zeros((mtx.shape[0], mtx.shape[1]))
    for i in range(mtx.shape[1]-1):
        col1 = mtx[i]
        col2 = mtx[i+1]
        distance = 0
        for j in range(len(col1)):
            math
    return new_mtx


if __name__ == "__main__":
    mtz = [
        [115, 58, 20],
        [10, 7, 11],
        [2, 0, 6],
        [0, 0, 38],
    ]

    mtz = numpy.array(wtf(mtz))
    n = mtz.shape[1]

    mtz = normalize(mtz)
    print(get_distance_matrix(mtz))
