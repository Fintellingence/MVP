from numba import float64, int32, njit, prange


@njit(int32(int32, float64[:], int32[:], float64))
def indexing_cusum(n, values, accum_ind, threshold):
    """
    Mark all indexes between which the sum of `values` exceed `threshold`
    Applied to slice arrays using cumulative sum of values

    Parameters
    ----------
    `n` : ``int``
        size of `values` and `accum_ind` array
    `values` : ``numpy.array(numpy.float64)``
        values to compute cumulative sum
    `threshold` : ``float``
        Value to reset cusum sweeping the `values` array and save index

    Output
    ------
    `accum_ind` : ``numpy.array(int32)``
        store indexes strides between which cusum exceeds the threshold
        Must have size `n` but only use all if all `values` are greater
        than `threshold`

    Return
    ------
    ``int``
        number of indexes marked (number of elements set in `accum_ind`)

    """
    cusum = 0.0
    accum_ind[0] = 0
    j = 1
    for i in prange(n):
        cusum = cusum + values[i]
        if cusum >= threshold:
            accum_ind[j] = i + 1
            j = j + 1
            cusum = 0.0
    return j


@njit(int32(int32, int32[:], int32[:]))
def indexing_new_days(n, days, new_days_ind):
    """
    Mark all indexes in which a new day begins from intraday array
    of day numbers. From a pandas time index in minutes scale, use
    `day` method and convert it to array of numpy integers. In all
    indexes `days` change, save in `new_days_ind`

    Parameters
    ----------
    `n` : ``int``
        size of `days` and `new_days` array
    `days` : ``numpy.array(numpy.int32)``
        days correponding to datetime index. Have `n` elements

    Output
    ------
    `new_days_ind` : ``numpy.array(numpy.int32)``
        store indexes in which new days begin. Must have size `n`
        Intraday data lies between indexes `new_days_ind[i]` and
        `new_days_ind[i + 1]`

    Return
    ------
    ``int``
        number of days transition found (effective size of `new_days_ind`)

    """
    new_days_ind[0] = 0
    j = 1
    for i in prange(n - 1):
        if days[i + 1] != days[i]:
            new_days_ind[j] = i + 1
            j = j + 1
    return j


@njit(int32(float64, float64, float64[:], int32, int32))
def binomial_series_converge(d, tolerance, w_array, w_size, last_index):
    """
    Compiled function to compute weights of fractional diff. efficiently
    It must be called until a positive number is returned, that indicate
    convergence was achieve according to `tolerance`

    The Binomial series for this problem is obtained expanding (1 - A)^d
    for any d real. The theoretical background is taken from

    [*] Advances in Financial Machine Learning, Marcos Lopez Prado, Wiley(2018)
        Chapter 5, section 5.4

    Parameters
    ----------
    `d` : ``float``
        order of fractional differentiation. Usually between 0 and 1
    `tolerance` : ``float``
        minimum value for weights to set cutoff in series expansion
    `w_array` : ``numpy.array``
        With weights computed up to `last_index`. Must have size `w_size`
    `w_size` : ``int``
        current size of `w_array`
    `last_index` : ``int``
        index of last weight set in `w_array` and from which must continue
        This means that up to index `last_index` all weights were computed

    Modified
    --------
    `w_array`
        with new weights up to `w_size` or the returned value depending on
        whether convergence was achied (last weight < `tolerance`)

    Return
    ------
    ``int``
        If positive, convergence was achieved and the value is the number
        of weights computed(smaller or equal than `w_size`). If negative,
        last weight is still above `tolerance` provided, the weights array
        must be resized adding empty entries. To achieve convergence this
        function must call again using `w_size - 1` as `last_index` after
        increasing `w_array` size

    """
    for k in prange(last_index + 1, w_size):
        w_array[k] = -(w_array[k - 1] / k) * (d - k + 1)
        if abs(w_array[k]) < tolerance:
            return k + 1
    return -1
