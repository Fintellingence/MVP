""" Module to analyzed refined data with basic statistical properties

    This module provide a workflow to extract directly from database some
    basic statistical properties from raw stock prices and volume. Before
    using it, have a look in `mvp.rawdata` documentation

    Class
    -----
    ``RefinedData``
        class that inherit ``RawData`` from mvp.rawdata to provide basic
        tools and statistical features to feed models though its methods
        It is designed to fill requirements of strategies as pointed in:

        https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/

        where some basic features as moving average and standard deviation
        are required not only using the default time-interval candlesticks
        but also over other targets, as volume and money bars.

"""
import numpy as np
import pandas as pd

from mvp import utils
from mvp.rawdata import RawData, assert_bar_type, assert_data_field


def assert_target(target):
    """
    Raise error if either `target` has invalid format or could
    not found the corresponding methods in `RawData` class
    """
    if len(target.split(":")) != 2:
        raise ValueError("Found invalid target '{}'".format(target))
    assert_bar_type(target.split(":")[0])
    assert_data_field(target.split(":")[1])


def assert_feature(feat_name):
    """
    Raise error if could not found feature `feat_name` method in
    ``RefinedData`` class. Use `available_features` if needed
    """
    method_name = "get_" + feat_name
    methods_set = set(RefinedData.__dict__.keys())
    if method_name not in methods_set:
        raise AttributeError(
            "Feature '{}' do not have '{}' method in RefinedData".format(
                feat_name, method_name
            )
        )


def available_features():
    """
    Return list of feature names available to use. Corresponds
    to all ``RefinedData`` methods starting with "get_"
    """
    methods_list = list(RefinedData.__dict__.keys())
    return [
        method_name.split("_")[1]
        for method_name in methods_list
        if method_name.split("_")[0] == "get"
    ]


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
        or index of 1-minute dataframe. Preferably ``pandas.Timestamp``
    `stop` : ``pandas.Timestamp`` or ``int``
        The final date/time instant to consider in computation
        or index of 1-minute dataframe. Preferably ``pandas.Timestamp``
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
    close prices from candlesticks built every time 10 million have
    been negotiated in the stock market

    `target` and `step` can also be used in the basic way, with
    time in minutes. If `target = "time:close"` and `step = 15`
    means the feature requested will be computed over close prices
    from candlesticks built after every 15 minuters of negotiation

    Some care is needed when dealing with non-time spaced target as
    an appropriate step will depend on the typical volume is traded
    for each specific company. For instance, the `step` values must
    have values according to company valuation

    Inherit
    -------
    ``mvp.rawdata.RawData``
        class to read databases and sample it in different formats

    """

    def __init__(
        self,
        symbol,
        db_path,
        preload={"time": [60, "day"], "money": None},
        requested_features={},
    ):
        """
        Initialize the class reading data from database. If only `symbol`
        and `db_path` are given simply initialize `RawData`. Some common
        bar types as well as features that will be exhaustively demanded
        can be previously loaded in this constructor

        Parameters
        ----------
        `symbol` : ``str``
            company symbol listed in stock market(available in database)
        `db_path` : ``str``
            full path to 1-minute database file
        `preload` : ``dict`` {`bar_type` : `step`}
            dictionary to inform dataframes to set in cache memory
            Available `bar_type` are given below while `step` must
            be ``int`` or ``list`` of integers. In case `bar_type`
            is "time", `step` also admit the string "day" and the
            integer values allowed are 1, 5, 10, 15, 30, 60
            Available bar types:
                "time"
                "tick"
                "volume"
                "money"
            Prefix of all ``RawData`` methods that end with `_bars`
        `requested_features` : ``dict {str, str}``
            Inform which features must be set in cache
            KEYS:
                The keys must be formated according to "bar_type:data_field"
                where `bar_type` inform how data bars are formed and
                `data_field` the bar value to use. Some examples are
                    "time:close"  - use close price in time-spaced bars
                    "time:volume" - use volume traded in time-spaced bars
                    "tick:high"   - use high price in tick-spaced bars
                    "money:close" - use close price in money-spaced bars
                This disctionary keys is also referred to as `target`
                parameters in this class methods to compute features
                The available names for `bar_type` are the same of
                the keys of `preload` parameter and are the methods
                of `RawData` that has as suffix `_bars`(also method
                of `RefinedData` as it inherit `RawData`)
                The available names for `data_field` are suffixes of
                any `RawData` method that starts with `get_`. Careful
                to do not confuse with the `RefinedData` methods that
                begins with `get_` which refers to features instead
            VALUES:
                String codifying all infomration to pass in methods call
                The values of this dictionaty must follow the convention
                "MET1_T1:V11,V12,...:MET2_T2:V21,V22,...:METM_TM:VM1,..."
                where MET is a `RefinedData` method suffix for all the
                ones that begins with `get_`. Therefore, available values
                to use can be consulted in `RefinedData` class methods
                Some (default) examples
                    "sma" = Moving Average (``int``)
                    "dev" = Standart Deviation (``int``)
                    "rsi" = Relative Strenght Index (RSI) indicator (``int``)
                    "fracdiff": Fractional differentiation (``float``)
                with the following data types of `Vij` in parentheses
                Note the underscore after METj which can be one of the
                following: 1, 5, 10, 15, 30, 60 and DAY indicating the
                time step to be used in bars size, in case the target
                provided in dict key is "time:*"
                In case other target is set, such as "money:*" any int
                is accepted, which is used to pack data in bars using
                cumulative sum of the referred target
                The values `Vij` must be of the type of the first
                argument(s)(required) of the feature method, that
                is one of those `RefinedData` methods with `get_`
                as prefix. Especifically for methods that require
                more than one argument, the syntax changes
                Example given dictionary key "time:*" with value:
                    "sma_60:100,1000:dev_DAY:10,20:autocorrmov_DAY:(20,5)"
                The moving average for 60-minute bars with windows of 100,
                1000, the moving standard deviation for daily bars with 10
                and 20 days, and finally the moving autocorrelation with
                daily bars for 20-days moving window and 5-days of shift
                will be set in cache
                Note that for `autocorrmov` the values are passed as tuple
                and are exactly used as `get_autocorrmov(*Vij, append=True)`
                For this reason, in this specific case, instead of using
                comma to separate the values(that are actually tuples), the
                user must use forward slashes '/'
                For instance: (20,5)/(200,20)
                WARNING:
                In this string no successive colon(:) is allowed as well as
                : at the end or beginning. Colons must aways be surrounded
                by keys and values, and this format will be checked before
                using for computations

        """
        RawData.__init__(self, symbol, db_path, preload)
        self.__cached_features = {}
        for target, inp_str in requested_features.items():
            assert_target(target)
            method_args_iter = utils.get_features_iterable(inp_str, target)
            for method_name, args, kwargs in method_args_iter:
                try:
                    self.__getattribute__(method_name)(*args, **kwargs)
                except AttributeError:
                    pass

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

    def get_volden(
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
            any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``

        """
        assert_target(target)
        bar_type = target.split(":")[0]
        str_code = self.__code_formatter("VOLDEN", step, 1, bar_type)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy().loc[start:stop]
        df = self.__getattribute__(bar_type + "_bars")(step=step)
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
            Any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``

        """
        assert_target(target)
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
            Any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``

        """
        assert_target(target)
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
            Any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``

        """
        assert_target(target)
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
            Any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``

        """
        assert_target(target)
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

    def get_autocorrperiod(
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
        compute the correlation between the two sets.

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
            Any further computation is avoided by a memory access

        Return
        ------
        ``float``

        """
        assert_target(target)
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

    def get_autocorrmany(
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
            Any further computation is avoided by a memory access

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

    def get_autocorrmov(
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
        See also `self.get_autocorrperiod`

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
            Any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``

        """
        assert_target(target)
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
        utils.numba_stats.moving_correlation(
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

    def __apply_fracdiff_weights(self, weights, x_vector):
        return np.dot(weights[::-1], x_vector)

    def get_fracdiff(
        self,
        d,
        start=None,
        stop=None,
        step=1,
        target="time:close",
        append=False,
        weights_tol=1e-3,
    ):
        """
        Compute fractional differentiation of a series with the binomial
        expansion formula for an arbitrary derivative order

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
            for convergence criterion. Limited to use a 1/3 of data pts
            or the target series

        Return
        ------
        ``pandas.Timeseries``

        """
        assert_target(target)
        dcode = "{:.2f}".format(d)
        str_code = self.__code_formatter("FRAC_DIFF", step, dcode, target)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].loc[start:stop]
        bar_type, field_name = target.split(":")
        field_method = self.__getattribute__("get_" + field_name)
        target_series = field_method(step=step, bar_type=bar_type)
        w = utils.fracdiff_weights(d, target_series.size // 3, weights_tol)
        fracdiff_series = target_series.rolling(window=w.size).apply(
            lambda x: self.__apply_fracdiff_weights(w, x), raw=True
        )
        fracdiff_series.name = "FracDiff"
        if append:
            self.__cached_features[str_code] = fracdiff_series.dropna()
        return fracdiff_series.loc[start:stop].dropna()

    def get_volafreq(
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
            Any further computation is avoided by a memory access

        Return
        ------
        `` pandas.Series ``

        Warning
        -------
        High demanding for large datasets (more than 10^4 data points)

        """
        assert_target(target)
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
            lambda x: utils.mean_quadratic_freq(x, step)
        )
        if append:
            self.__cached_features[str_code] = vol_series.dropna()
        return vol_series.loc[start:stop].dropna()

    def get_volagain(
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
            Any further computation is avoided by a memory access

        Return
        ------
        ``pandas.Series``
            Maximum profit/loss divided by moving average series

        """
        assert_target(target)
        bar_type = target.split(":")[0]
        str_code = self.__code_formatter("VOLA_GAIN", step, window, bar_type)
        start, stop = self.assert_window(start, stop)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].loc[start:stop]
        df = self.__getattribute__(bar_type + "_bars")(step=step)
        max_price = df.High.rolling(window).max()
        min_price = df.Low.rolling(window).min()
        ma = self.get_sma(window, step=step, target=bar_type + ":close")
        vol_series = (max_price - min_price) / ma
        vol_series.name = "GainVolatility"
        if append:
            self.__cached_features[str_code] = vol_series.dropna()
        return vol_series.loc[start:stop].dropna()
