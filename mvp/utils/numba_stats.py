from math import sqrt

from numba import float64, int32, njit, prange


@njit(float64(int32, float64[:]))
def average(pts, data_set):
    """ Return average of the first `pts` data points of `data_set` """
    sum_set = 0.0
    for i in prange(pts):
        sum_set = sum_set + data_set[i]
    return sum_set / pts


@njit(float64(int32, float64[:]))
def square_res_sum(pts, data_set):
    """
    Return sum of square residuals from the average
    for the first `pts` points of `data_set`
    """
    sum_sq = 0.0
    set_avg = average(pts, data_set)
    for i in prange(pts):
        sum_sq = sum_sq + (data_set[i] - set_avg) ** 2
    return sum_sq


@njit(float64(int32, float64[:], float64[:]))
def correlation(pts, data_set1, data_set2):
    """
    Compute standard correlation between two data sets

    Parameters
    ---
    `pts` : ``int``
        Number of points to consider starting from index 0
        Maximum value is min(data_set1.size, data_set2.size)
    `data_set1` : ``numpy.array[float64]``
    `data_set2` : ``numpy.array[float64]``

    """
    mutual_dev = 0.0
    sum_square_dev1 = 0.0
    sum_square_dev2 = 0.0
    set1_avg = average(pts, data_set1)
    set2_avg = average(pts, data_set2)
    for i in prange(pts):
        data_dev1 = data_set1[i] - set1_avg
        data_dev2 = data_set2[i] - set2_avg
        mutual_dev = mutual_dev + data_dev1 * data_dev2
        sum_square_dev1 = sum_square_dev1 + data_dev1 ** 2
        sum_square_dev2 = sum_square_dev2 + data_dev2 ** 2
    standard_dev1 = sqrt(sum_square_dev1)
    standard_dev2 = sqrt(sum_square_dev2)
    return mutual_dev / standard_dev1 / standard_dev2


@njit((int32, float64[:], float64[:], float64[:]))
def moving_correlation(window, data_set1, data_set2, mv_corr):
    """
    Compute correlation using moving `window` data points
    All arrays are required to have the same size despite
    the output array `mv_corr` waste the first `(window - 1)` elements
    All arrays must have the same size

    Parameters
    ---
    `window` : ``int``
        number of sequential data points to use along the data sets
    `data_set1` : ``numpy.array[float64]``
    `data_set2` : ``numpy.array[float64]``

    Output Parameter
    ---
    `mv_corr` : ``numpy.array[float64]``
        mv_corr[i + window - 1] = (
            correlation(data_set1[i : window], data_set2[i : window])
        )
        with i = 0, 1, ..., data_set.size

    Warning
    ---
    The algorithm is implemented using a progressive iterative method
    where in the looping moving the `window` along the data sets all
    required calculations use the results of previous windows. For
    example to update the average we just need to add the contribution
    of the new data point in the far right and remove the data point
    in the far left. As result this boost the performance but is prone
    to numerical error for very large data sets. The error can be
    estimated by the data sets size as the number of decimal places
    lost in precision.

    """
    set_size = data_set1.size
    if window > set_size:
        return
    corr = correlation(window, data_set1, data_set2)
    avg1 = average(window, data_set1)
    avg2 = average(window, data_set2)
    sig1 = square_res_sum(window, data_set1)
    sig2 = square_res_sum(window, data_set2)
    mv_corr[window - 1] = corr
    for i in prange(set_size - window):
        j = i + window
        old_avg1 = avg1
        old_avg2 = avg2
        cross_sum = sqrt(sig1 * sig2) * corr + window * avg1 * avg2
        avg1 = avg1 + (data_set1[j] - data_set1[i]) / window
        avg2 = avg2 + (data_set2[j] - data_set2[i]) / window
        sig1 = (
            sig1
            - data_set1[i] ** 2
            + data_set1[j] ** 2
            + window * (old_avg1 ** 2 - avg1 ** 2)
        )
        sig2 = (
            sig2
            - data_set2[i] ** 2
            + data_set2[j] ** 2
            + window * (old_avg2 ** 2 - avg2 ** 2)
        )
        new_cross_sum = (
            cross_sum
            - data_set1[i] * data_set2[i]
            + data_set1[j] * data_set2[j]
        )
        if (sig1 * sig2 == 0):
            mv_corr[j] = 0
        else:
            corr = (new_cross_sum - window * avg1 * avg2) / sqrt(sig1 * sig2)
            mv_corr[j] = corr
