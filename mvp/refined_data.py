from math import sqrt

import numpy as np
import pandas as pd
from numba import float64, int32, njit, prange

import numba_stats
from mvp.rawdata import RawData


@njit(int32(float64, float64, float64[:], int32, int32))
def numba_weights(d, tolerance, w_array, w_size, last_index):
    """
    Compiled function to compute weights of frac_diff method efficiently
    This function must be called until a positive number is returned,
    which indicate that convergence was achieve according to `tolerance`

    Parameters
    ----------
    `d` : ``float``
        order of fractional differentiation. Usually between 0 and 1
    `tolerance` : ``float``
        minimum value for weights
    `w_array` : ``numpy.array``
        values of all weights computed so far
    `w_size` : ``int``
        current size of `w_array`
    `last_index` : ``int``
        index of last weight set in `w_array` and from which must continue

    Modified
    --------
    `w_array`
        with new weights starting from `last_index`

    Return
    ------
    ``int``
        If positive, convergence was achieved and the value is the number
        of weights computed. If negative, weights are above the `tolerance`
        provided, the weights array must be resized adding empty entries,
        and this function must be called again from the last weight set

    """
    for k in prange(last_index + 1, w_size):
        w_array[k] = -(w_array[k - 1] / k) * (d - k + 1)
        if abs(w_array[k]) < tolerance:
            return k + 1
    return -1


def mean_quadratic_freq(data, time_spacing=1):
    """
    Use `data` Fourier transform to compute mean quadratic freq

    Parameters
    ----------
    `data` : ``numpy.array``
        data series in time basis
    `time_spacing` : ``float``
        sample time spacing among data points

    Return
    ------
    ``float``

    """
    displace_data = data - data.mean()
    fft_weights = np.fft.fft(displace_data)
    freq = np.fft.fftfreq(displace_data.size, time_spacing)
    weights_norm = np.abs(fft_weights).sum()
    return sqrt((freq * freq * np.abs(fft_weights)).sum() / weights_norm)


class RefinedData(RawData):
    """
    Integrate to the raw data with open-high-low-close values
    some simple statistical features which provide more tools
    to analyze data and support models

    Most part of attributes depend on a set of common parameters
    which are related to time period, bar type and size. A brief
    explanation is provided below

    `start` : ``pandas.Timestamp`` or ``int``
        The initial date/time instant to consider in computation
        or index of dataframe. Preferably ``pandas.Timestamp``
    `stop` : ``pandas.Timestamp`` or ``int``
        The final date/time instante to consider in computation
        or index of dataframe. Preferably ``pandas.Timestamp``
    `step` : ``int`` (especially string "day")
        Dataframe bar's spacing value according to `target`. The
        values can change drastically depending on `target`, see
        some examples below
    `target` : ``str``
        String fromatted as "bar_type:data_field". Parameters of
        this pair are
            bar_type : ["time", "tick", "volume", "money"]
            data_field : ["open", "high", "low", "close", "volume"]

    Consider the following example: if `target = "money:close"` and
    `step = 10000000` the feature requested will be computed over
    close prices from candlesticks built after 10 million have been
    negotiated in the stock market.

    Another usage of `target` and `step` parameter is the basic way
    the candlesticks are used. If `target = "time:close"` and
    `step = 15` means the feature requested will be computed over
    close prices from candlesticks built after every 15 minuters of
    negotiation in the stock market

    Inherit
    -------
    ``mvp.rawdata.RawData``
        class to read databases and sample it in different formats

    """

    def __init__(
        self,
        symbol,
        db_path,
        preload={"time": [60, "day"]},
        requested_features={},
        step=1,
        target="time:close",
    ):
        """
        Initialize the class reading data from database. If only `symbol`
        and `db_path` are given simply initialize `RawData`. Some common
        features may be computed in initialization which are NECESSARILY
        set in cache memory to avoid computing effort in next calls

        Parameters
        ----------
        `symbol` : ``str``
            company symbol code listed in stock market
        `db_path` : ``str``
            full path to 1-minute database file
        `preload` : ``dict`` {`bar_type` : `step`}
            dictionary to inform dataframes to set in cache memory
            Available `bar_type` are given below while `step` must
            be ``int`` or ``list`` of integeres. In case `bar_type`
            is "time", `step` also admit the string "day"
            Available bar types:
                "time"
                "tick"
                "volume"
                "money"
        `requested_features` : ``dict`` {`feature_name` : `parameter`}
            Dictionary codifying some features to compute in initialization
            Available values of `feature_name`:`parameter` pairs
                "sma" : ``int``
                "dev" : ``int``
                "rsi" : ``int``
                "returns" : ``int``
                "vol_den" : ()
                "fracdiff" : ``int``
                "autocorr_mov" : (``int``, ``int``)
                "vola_freq" : ``int``
                "vola_gain" : ``int``
            Where `parameter` also accepts list of types listed above
        `step` : ``int`` or "day"
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]

        """
        RawData.__init__(self, symbol, db_path, preload)
        self.__cached_features = {}
        extra_args = {"step": step, "target": target, "append": True}
        for feature, params in requested_features.items():
            feature = feature.lower()
            method_str = "get_" + feature
            if not hasattr(self, method_str):
                print("Method {} not found in RefinedData".format(method_str))
                continue
            method = self.__getattribute__(method_str)
            if not isinstance(params, list):
                params = [params]
            for param in params:
                try:
                    if isinstance(param, tuple):
                        method(*param, **extra_args)
                    else:
                        method(param, **extra_args)
                except Exception as err:
                    print("\nError in {}\n\n".format(method_str), err)

    def __code_formatter(self, name, step, extra_par, target):
        """ Format string to use as key for cache dictionary """
        return "{}_{}_{}_{}".format(name, step, extra_par, target)

    def cache_features_keys(self):
        """ internal keys used to cached features. Information purposes """
        return list(self.__cached_features.keys())

    def cache_features_size(self):
        """ Total size of cached features in bytes """
        full_size = 0
        for feature_data in self.__cached_features.values():
            full_size = full_size + feature_data.__sizeof__()
        return full_size

    def cache_clean(self):
        """ delete all features in cache memory """
        del self.__cached_features
        self.__cached_features = {}

    def get_vol_den(
        self, start=None, stop=None, step=1, target="time:close", append=False
    ):
        """
        Average volume exchange per tick/deal

        Parameters
        ----------
        `start` : ``pandas.Timestamp``
            initial time instant to slice the data timeseries
        `stop` : ``pandas.Timestamp``
            final time instant to slice the data timeseries
        `step` : ``int``
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name`, is not applicable in this feature
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``

        """
        str_code = self.__code_formatter("MA", step, 1, target)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy().loc[start:stop]
        bar_method = self.__getattribute__(target.split(":")[0] + "_bars")
        df = bar_method(step=step)
        vol_den = df.Volume / df.TickVol
        clean_data = vol_den.replace([-np.inf, np.inf], np.nan).copy()
        clean_data.dropna(inplace=True)
        clean_data.name = "VolumeDensity"
        if append:
            self.__cached_features[str_code] = clean_data.astype(int)
        return clean_data[start:stop].astype(int)

    def get_sma(
        self,
        window,
        start=None,
        stop=None,
        step=1,
        target="time:close",
        append=False,
    ):
        """
        Compute simple moving average(sma)

        Parameters
        ----------
        `window` : ``int``
            number of data points to consider in each evaluation
        `start` : ``pandas.Timestamp``
            initial time instant to slice the data timeseries
        `stop` : ``pandas.Timestamp``
            final time instant to slice the data timeseries
        `step` : ``int``
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]. Consult this
            class `RefinedData` documentation for more info
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``
            simple moving average

        """
        str_code = self.__code_formatter("MA", step, window, target)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy().loc[start:stop]
        bar_type, field_name = target.split(":")
        field_method = self.__getattribute__("get_" + field_name)
        target_series = field_method(step=step, bar_type=bar_type)
        moving_avg = target_series.rolling(window).mean()
        moving_avg.name = "MovingAverage"
        if append:
            self.__cached_features[str_code] = moving_avg.dropna()
        return moving_avg[start:stop].dropna()

    def get_dev(
        self,
        window,
        start=None,
        stop=None,
        step=1,
        target="time:close",
        append=False,
    ):
        """
        Compute moving standard deviation(dev)

        Parameters
        ----------
        `window` : ``int``
            number of data points to consider in each evaluation
        `start` : ``pandas.Timestamp``
            initial time instant to slice the data timeseries
        `stop` : ``pandas.Timestamp``
            final time instant to slice the data timeseries
        `step` : ``int``
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]. Consult this
            class `RefinedData` documentation for more info
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``
            moving standard deviation

        """
        str_code = self.__code_formatter("DEV", step, window, target)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy().loc[start:stop]
        bar_type, field_name = target.split(":")
        field_method = self.__getattribute__("get_" + field_name)
        target_series = field_method(step=step, bar_type=bar_type)
        moving_std = target_series.rolling(window).std()
        moving_std.name = "StandardDeviation"
        if append:
            self.__cached_features[str_code] = moving_std.dropna()
        return moving_std.loc[start:stop].dropna()

    def get_returns(
        self,
        window=1,
        start=None,
        stop=None,
        step=1,
        target="time:close",
        append=False,
    ):
        """
        Compute normalized returns of buy operations executed in a window

        Parameters
        ----------
        `window` : ``int``
            number of data points to consider in each evaluation
        `start` : ``pandas.Timestamp``
            initial time instant to slice the data timeseries
        `stop` : ``pandas.Timestamp``
            final time instant to slice the data timeseries
        `step` : ``int``
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]. Consult this
            class `RefinedData` documentation for more info
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``
            normalized returns

        """
        str_code = self.__code_formatter("RET", step, window, target)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy().loc[start:stop]
        bar_type, field_name = target.split(":")
        field_method = self.__getattribute__("get_" + field_name)
        target_series = field_method(step=step, bar_type=bar_type)
        return_series = (
            target_series - target_series.shift(window)
        ) / target_series.shift(window)
        return_series.name = "ReturnSeries"
        if append:
            self.__cached_features[str_code] = return_series.dropna()
        return return_series.loc[start:stop].dropna()

    def get_rsi(
        self,
        window,
        start=None,
        stop=None,
        step=1,
        target="time:close",
        append=False,
    ):
        """
        Return moving Relative Strength Index time series

        Parameters
        ----------
        `window` : ``int``
            number of data points to consider in each evaluation
        `start` : ``pandas.Timestamp``
            initial time instant to slice the data timeseries
        `stop` : ``pandas.Timestamp``
            final time instant to slice the data timeseries
        `step` : ``int``
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]. Consult this
            class `RefinedData` documentation for more info
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``

        """
        str_code = self.__code_formatter("RSI", step, window, target)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy().loc[start:stop]
        bar_type, field_name = target.split(":")
        field_method = self.__getattribute__("get_" + field_name)
        target_series = field_method(step=step, bar_type=bar_type)
        return_series = target_series - target_series.shift(periods=1)
        gain_or_zero = return_series.apply(lambda x: 0 if x < 0 else x)
        loss_or_zero = return_series.apply(lambda x: 0 if x > 0 else -x)
        ratio = (
            gain_or_zero.rolling(window=window).mean().dropna()
            / loss_or_zero.rolling(window=window).mean().dropna()
        )
        rsi_series = ratio.apply(lambda x: 100 - 100 / (1 + x))
        rsi_series.name = "RelativeStrengthIndex"
        if append:
            self.__cached_features[str_code] = rsi_series.dropna()
        return rsi_series.loc[start:stop].dropna()

    def get_autocorr_period(
        self,
        shift,
        start=None,
        stop=None,
        step=1,
        target="time:close",
        append=False,
    ):
        """
        Compute auto-correlation function in a time period
        According to pandas the autocorrelation is computed as follows:
        Given a set {y1, y2, ... , yN} divide it in {y1, y2, ..., yN-s}
        and {ys, ys+1, ..., yN}, compute the average for each set, then
        compute the covariance between the two sets.

        Parameters
        ----------
        `shift` : ``int``
            displacement to separate the two data samples
        `start` : ``pandas.Timestamp`` or ``int``
            first dataframe index of the period
        `stop` : ``pandas.Timestamp`` or ``int``
            last dataframe index of the period
        `step` : ``int``
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]. Consult this
            class `RefinedData` documentation for more info
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access

        Return
        ------
        ``float``

        """
        start, stop = self.assert_window(start, stop)
        extra_code = "{}_{}_{}".format(shift, start, stop)
        str_code = self.__code_formatter("AUTOCORR", step, extra_code, target)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        bar_type, field_name = target.split(":")
        field_method = self.__getattribute__("get_" + field_name)
        target_series = field_method(start, stop, step, bar_type)
        if shift >= target_series.size:
            raise ValueError(
                "Period enclosed from {} to {} provided {} "
                "data points, while {} shift was required".format(
                    start, stop, target_series.size, shift
                )
            )
        autocorr = target_series.autocorr(lag=shift)
        if append:
            self.__cached_features[str_code] = autocorr
        return autocorr

    def get_autocorr_many(
        self,
        shift,
        starts,
        stops,
        step=1,
        target="time:close",
        append=False,
    ):
        """
        Compute auto-correlation function for multiple time intervals
        For each start and stop call `self.autocorr_period`

        Parameters
        ----------
        `shift` : ``int``
            displacement to separate the two data samples
        `starts` : ``list`` of ``pandas.Timestamp``
            list of first date/minute of the period. Size of `stops`
        `stops` : ``list`` of ``pandas.Timestamp``
            list of end of the period. Size of `starts`
        `step` : ``int``
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]. Consult this
            class `RefinedData` documentation for more info
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access

        Return
        ------
        ``pandas.DataFrame``
            Several results of autocorrelation in different periods

        """
        if len(starts) != len(stops):
            raise ValueError("starts and stops must have same size")
        autocorr = np.empty(len(starts))
        for i, (start, stop) in enumerate(zip(starts, stops)):
            try:
                autocorr[i] = self.get_autocorr_period(
                    shift, start, stop, step, target, append
                )
            except Exception:
                autocorr[i] = np.nan
        autocorr_df = pd.DataFrame(
            {"FinalDate": stops, "Autocorr": autocorr}, index=starts
        )
        autocorr_df.index.name = "InitialDate"
        return autocorr_df

    def get_autocorr_mov(
        self,
        window,
        shift,
        start=None,
        stop=None,
        step=1,
        target="time:close",
        append=False,
    ):
        """
        Compute auto-correlation in a moving `window`

        Parameters
        ----------
        `window` : ``int``
            size of moving window. Must be larger than `shift`
        `shift` : ``int``
            displacement to separate the two data samples in moving window
        `start` : ``pd.Timestamp`` or ``int``
            First index/date. Default is the beginning of dataframe
        `stop` : ``pd.Timestamp`` or ``int``
            Last index/date. Default is the end of dataframe
        `step` : ``int``
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]. Consult this
            class `RefinedData` documentation for more info
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``

        """
        if shift >= window:
            raise ValueError(
                "The shift between two data sets {} must be smaller "
                "than the moving window {}".format(shift, window)
            )
        start, stop = self.assert_window(start, stop)
        par_code = "{}_{}".format(window, shift)
        str_code = self.__code_formatter(
            "MOV_AUTOCORR", step, par_code, target
        )
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy().loc[start:stop]
        bar_type, field_name = target.split(":")
        field_method = self.__getattribute__("get_" + field_name)
        target_series = field_method(step=step, bar_type=bar_type)
        target_vals = target_series.values
        if target_vals.size < window:
            raise ValueError(
                "The number of data points between {} and {} "
                "is {} while window of size {} was required".format(
                    start, stop, target_vals.size, window
                )
            )
        pts = target_vals.size
        corr_window = window - shift
        lag_series = target_vals[: pts - shift]
        adv_series = target_vals[shift:]
        mov_autocorr = np.zeros(pts - shift)
        numba_stats.moving_correlation(
            corr_window, lag_series, adv_series, mov_autocorr
        )
        core_data = mov_autocorr[corr_window - 1 :]
        mov_autocorr_ser = pd.Series(
            core_data, index=target_series.index[window - 1 :]
        )
        mov_autocorr_ser.name = "Autocorrelation"
        if append:
            self.__cached_features[str_code] = mov_autocorr_ser
        return mov_autocorr_ser.loc[start:stop]

    def __frac_diff_weights(self, d, tolerance, max_weights=1e8):
        """
        Compute weights of frac_diff binomial-expansion formula

        Parameters
        ----------
        `d` : ``float``
            order of fractional differentiation. Usually between 0 and 1
        `tolerance` : ``float``
            minumum acceptable value for weights to compute in series
        `max_weights` : ``int``
            max number of weights (to avoid excessive memory consumption)

        Return
        ------
        ``numpy.array``
            wegiths/coefficients of binomial series expansion

        """
        w_array = np.empty(100)
        w_array[0] = 1.0
        flag = -1
        last_i = 0
        while flag < 0:
            flag = numba_weights(d, tolerance, w_array, w_array.size, last_i)
            if w_array.size > max_weights:
                print(
                    "WARNING : could not achieved required weights "
                    "accuracy in frac_diff. Last weight = {}".format(
                        w_array[-1]
                    )
                )
                return w_array
            if flag < 0:
                last_i = w_array.size - 1
                w_array = np.concatenate([w_array, np.empty(10 * last_i)])
        return w_array[:flag]

    def __apply_weights(self, weights, x_vector):
        return np.dot(weights[::-1], x_vector)

    def get_fracdiff(
        self,
        d,
        start=None,
        stop=None,
        step=1,
        target="time:close",
        append=False,
        weights_tol=1e-5,
    ):
        """
        Compute fractional differentiation of a series with the binomial
        expansion formula for an arbitrary derivative order.

        Parameters
        ----------
        `d` : ``float``
            derivative order (d = 1 implies daily returns) lying in (0,1]
        `start` : ``pandas.Timestamp`` or ``int``
            first dataframe index of the period
        `stop` : ``pandas.Timestamp`` or ``int``
            last dataframe index of the period
        `step` : ``int``
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]. Consult this
            class `RefinedData` documentation for more info
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access
        `weights_tol` : ``float``
            minimum value for a weight in the binomial series expansion

        Return
        ------
        ``pandas.Timeseries``

        """
        str_code = self.__code_formatter("FRAC_DIFF", step, d, target)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].loc[start:stop]
        bar_type, field_name = target.split(":")
        field_method = self.__getattribute__("get_" + field_name)
        target_series = field_method(step=step, bar_type=bar_type)
        w = self.__frac_diff_weights(d, weights_tol)
        fracdiff_series = target_series.rolling(window=w.size).apply(
            lambda x: self.__apply_weights(w, x), raw=True
        )
        fracdiff_series.name = "FracDiff"
        if append:
            self.__cached_features[str_code] = fracdiff_series.dropna()
        return fracdiff_series.loc[start:stop].dropna()

    def get_vola_freq(
        self,
        window,
        start=None,
        stop=None,
        step=1,
        target="time:close",
        append=False,
    ):
        """
        Introduce/compute a measure of volatility based on how rapidly
        the stock prices are oscillating around the average or if the
        series presents a narrow peak. Use the Fourier representation
        of the time series to compute the mean quadratic frequency of
        the spectrum. Note that in case the mean frequency is zero this
        is just the frequency standard deviation.

        Parameters
        ----------
        `window` : ``int``
            number of data points to consider in each evaluation
        `start` : ``pandas.Timestamp``
            initial time instant to slice the data timeseries
        `stop` : ``pandas.Timestamp``
            final time instant to slice the data timeseries
        `step` : ``int``
            dataframe bar's spacing value according to `target`
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]. Consult this
            class `RefinedData` documentation for more info
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access

        Return
        ------
        `` pandas.Series ``
            Each Timestamp index contains the mean quadratic frequency.

        Warning
        -------
        High demanding for large datasets (more than 10^4 data points)

        """
        str_code = self.__code_formatter("VOLA_FREQ", step, window, target)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].loc[start:stop]
        bar_type, field_name = target.split(":")
        field_method = self.__getattribute__("get_" + field_name)
        target_series = field_method(step=step, bar_type=bar_type)
        if isinstance(step, str):
            step = 1
        vol_series = target_series.rolling(window).apply(
            lambda x: mean_quadratic_freq(x, step)
        )
        if append:
            self.__cached_features[str_code] = vol_series.dropna()
        return vol_series.loc[start:stop].dropna()

    def get_vola_gain(
        self,
        window,
        start=None,
        stop=None,
        step=1,
        target="time:close",
        append=False,
    ):
        """
        Profit of the best possible trade divide by its mean value
        in a moving window, that is, the best profit relative to
        its moving average. Note this is also the absolute value
        of the maximum loss in the period.

        Parameters
        ----------
        `window` : ``int``
            number of data points to consider in each evaluation
        `start` : ``pandas.Timestamp``
            initial time instant to slice the data timeseries
        `stop` : ``pandas.Timestamp``
            final time instant to slice the data timeseries
        `step` : ``int``
            accumulated amount of dataframe field to form new bar
        `target` : ``str``
            string in the form "bar_type:field_name". The first part
            `bar_type` refers to which quantity `step` refers to, as
            ["time", "tick", "volume", "money"]. The second part,
            `field_name` refers to one of the values in candlesticks
            ["open", "high", "low", "close", "volume"]. Consult this
            class `RefinedData` documentation for more info
        `append` : ``bool``
            Whether to append the full time series in cache memory or not
            The all-time data series is used in `append=True` case as for
            any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``
            Maximum profit/loss divided by moving average series

        """
        str_code = self.__code_formatter("VOLA_GAIN", step, window, target)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].loc[start:stop]
        bar_type = target.split(":")[0]
        df = self.__getattribute__(bar_type + "_bars")(step=step)
        max_price = df.High.rolling(window).max()
        min_price = df.Low.rolling(window).min()
        ma = self.get_sma(window, step=step, target=bar_type + ":close")
        vol_series = (max_price - min_price) / ma
        vol_series.name = "GainVolatility"
        if append:
            self.__cached_features[str_code] = vol_series.dropna()
        return vol_series.loc[start:stop].dropna()
