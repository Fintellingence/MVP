import numpy as np
from math import sqrt
from mvp.utils import binomial_series_converge


def mean_quadratic_freq(data, time_spacing=1):
    """
    Use `data` Fourier transform to compute mean quadratic freq
    from fluctuations around the average

    Parameters
    ----------
    `data` : ``numpy.array``
        data array
    `time_spacing` : ``float``
        sample time spacing among data points

    Return
    ------
    ``float``
        Mean quadratic frequency

    """
    displace_data = data - data.mean()
    fft_weights = np.fft.fft(displace_data)
    freq = np.fft.fftfreq(displace_data.size, time_spacing)
    weights_norm = np.abs(fft_weights).sum()
    return sqrt((freq * freq * np.abs(fft_weights)).sum() / weights_norm)


def fracdiff_weights(d, max_weights=100, tol=None):
    """
    Compute weights of fractional differentiation binomial series
    See also `mvp.utils.binomial_series_converge` and theoretical
    background in

    [*] Advances in Financial Machine Learning, Marcos Lopez Prado, Wiley(2018)
        Chapter 5, section 5.4

    Parameters
    ----------
    `d` : ``float``
        order of fractional differentiation. Usually between 0 and 1
    `max_weights` : ``int``
        max number of weights to limit data/memory consumption
    `tol` : ``float``
        minumum acceptable value for weights to automatic series cutoff
        In convergence process if exceed `max_weights` print a warning,
        but return the array without errors

    Return
    ------
    ``numpy.array``
        wegiths/coefficients of binomial series expansion

    """
    if tol is None:
        w = np.ones(max_weights)
        binomial_series_converge(d, tol, w, w.size, 0)
        return w
    step = 1 + max_weights // 10
    w = np.empty(step)
    w[0] = 1.0
    flag = -1
    last = 0
    while flag < 0:
        flag = binomial_series_converge(d, tol, w, w.size, last)
        if w.size > max_weights:
            print(
                "[!] Binomial series stiff cutoff by "
                "max_weights {}".format(max_weights)
            )
            return w[:max_weights]
        last = w.size - 1
        w = np.concatenate([w, np.empty(step)])
    return w[:flag]
