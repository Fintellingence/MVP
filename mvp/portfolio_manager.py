""" Portfolio module

This module provide statistical features of a set of companies
and implement basic tools to obtain portfolio performance. The
main goal is to provide tools for dynamic portfolio management

Classes
-------

``RefinedSet(db_path -> str, preload -> dict, cache_common_features -> dict)``

``StockDeal(
    symbol -> str,
    quantity -> int,
    unit_price -> float,
    date -> pandas.Timestamp,
    flag -> int,
    fixed_tax -> float,
    relative_tax -> float,
    daily_tax -> float,
)``

``PortfolioRecord()``

``Portfolio(
    db_path -> str,
    fixed_tax -> float,
    relative_tax -> float,
    preload -> dict,
    common_features -> dict,
)``

"""

import os
import bisect
import numpy as np
import pandas as pd

from functools import total_ordering
from mvp.refined_data import RefinedData, assert_target, assert_feature
from mvp.utils import numba_stats
from mvp.utils import validate_features_string, get_features_iterable
from mvp.labels import event_label


class RefinedSet:
    """
    Class to collect a set of stock shares and obtain common features
    Provide methods to dynamically insert and remove symbols

    Main attribute
    --------------
    `refined_obj` : ``dict {str : RefinedData}``
        set of refined data objects accessible using the symbol as key
    """

    def __init__(
        self,
        db_path,
        preload={},
        cache_common_features={},
    ):
        """
        Parameters
        ----------
        `db_path` : ``str``
            full path to 1-minute database file
        `preload` : ``dict {str : int}``
            Inform dataframes to be set in cache for fast access
            Available keys are methods of `RawData` class, which
            end with the suffix `_bars`, and the value is `step`
            argument required in these methods. Some examples
            {
                "time" : ``int`` or "day"
                "tick" : ``int``
                "volume" : ``int``
                "money" : ``int``
            }
            The value field also support ``list`` of these types
            See also ``mvp.rawdata.RawData`` class documentation
        `cache_common_features` : ``dict {str : str}``
            dictionary to preload set of common features for all symbols
            introduced in this `RefinedSet` through `RefinedData` object
            This is a quite complicated optional argument because of its
            specific improvement role. See a more detailed documentation
            in ``mvp.refined_data.RefinedData`` class constructor

        Warning
        -------
        The parameters `preload` and `cache_common_features` may slow down
        the inclusion of new symbols, due to calculations required to have
        in class cache memory
        """
        if not os.path.isfile(db_path):
            raise IOError("Database file {} not found".format(db_path))
        self.db_path = db_path
        self.preload_data = preload
        self.cache_common_features = {}
        for target, data_string in cache_common_features.items():
            assert_target(target)
            self.cache_common_features[target] = validate_features_string(
                data_string, target.split(":")[0]
            )
        self.refined_obj = {}
        self.symbol_period = {}

    def __clean_features_cache(self):
        """For each refined object in this set clean the cache"""
        for ref_obj in self.refined_obj.values():
            ref_obj.cache_clean()

    def is_empty(self):
        """
        Return `True` if there are no symbols refined in this object
        """
        return not self.refined_obj

    def has_symbol(self, symbol):
        """Return True if the `symbol` (or list of) is in the set"""
        if not isinstance(symbol, list):
            return symbol in self.refined_obj.keys()
        return set(symbol).issubset(set(self.refined_obj.keys()))

    def memory_comsumption(self):
        """
        return the total memory (approximately) being used in bytes
        """
        total_size = 0
        for ref_obj in self.refined_obj.values():
            feat_size = ref_obj.cache_features_size()
            df_size = ref_obj.cache_dataframes_size()
            total_size = total_size + feat_size + df_size
        return total_size

    def display_refined_info(self):
        """
        Print on screen current status of this symbol-set object
        """
        print("\nActual status of refined set\n")
        for symbol in self.refined_obj.keys():
            start = self.symbol_period[symbol][0]
            stop = self.symbol_period[symbol][1]
            print("{} from {} to {}".format(symbol, start, stop))
        print("\nFeatures requested to be in cache memory")
        print(self.cache_common_features)

    def add_common_features(self, new_cache_features):
        """
        Append `new_cache_features` to set in cache for all symbols

        Parameters
        ----------
        `new_cache_features` : ``dict {str : str} ``
            Features to automatically set in cache in every `RefinedData`
            For more info on how to format dict keys and values see this
            class or `RefinedData` constructor documentation

        Warning
        -------
        All features are refreshed and may take some time depending
        on the number of company symbols currently in the set
        """
        if not isinstance(new_cache_features, dict):
            print("New features must be informed as dictionary. Aborted")
            return
        for target, data_string in new_cache_features.items():
            bar_type = target.split(":")[0]
            if target in self.cache_common_features.keys():
                new_str = (
                    self.cache_common_features[target] + ":" + data_string
                )
            else:
                new_str = data_string
            try:
                valid_new_string = validate_features_string(new_str, bar_type)
                self.cache_common_features[target] = valid_new_string
            except ValueError as err:
                print("Target '{}' raised the error:".format(target), err)
        self.refresh_all_features()

    def reset_common_features(self, new_cache_features):
        """
        Reset all features to set in cache previously defined
        To have an empty cache just pass a empty dictionary

        Parameters
        ----------
        `new_cache_features` : ``dict {str : str} ``
            Features to automatically set in cache in every `RefinedData`
            For more info on how to format dict keys and values see this
            class or `RefinedData` constructor documentation

        Warning
        -------
        All features are refreshed and may take some time depending
        on the number of company symbols currently in the set
        """
        if not isinstance(new_cache_features, dict):
            print("New features must be informed as dictionary. Aborted")
            return
        self.cache_common_features = {}
        for target, data_string in new_cache_features.items():
            bar_type = target.split(":")[0]
            try:
                self.cache_common_features[target] = validate_features_string(
                    data_string, bar_type
                )
            except ValueError as err:
                print("Target '{}' raised the error:".format(target), err)
        self.refresh_all_features()

    def new_refined_symbol(self, symbol, start=None, stop=None):
        """
        Introduce new symbol in the set for a given period

        Parameters
        ----------
        `symbol` : ``str``
            valid symbol in the `self.db_path` database
        `start` : ``pandas.Timestamp``
            datetime of inclusion in the set
        `stop` : ``pandas.Timestamp``
            datetime of exclusion in the set
        """
        if symbol in self.refined_obj.keys():
            return
        self.refined_obj[symbol] = RefinedData(
            symbol, self.db_path, self.preload_data, self.cache_common_features
        )
        valid_start, valid_stop = self.refined_obj[symbol].assert_window(
            start, stop
        )
        self.symbol_period[symbol] = (valid_start, valid_stop)

    def remove_refined_symbol(self, symbol):
        """
        Remove company symbol from the set. Use `dict.pop` method

        Parameters
        ----------
        `symbol` : ``str``
            valid company symbol previous initialized

        Return
        ------
        `RefinedData` or `None`
            If the company symbol were in the set, return its `RefinedData`
        """
        self.symbol_period.pop(symbol, None)
        return self.refined_obj.pop(symbol, None)

    def refresh_all_features(self):
        """
        Compute again all common features in `self.cache_common_features`
        """
        for target, inp_str in self.cache_common_features.items():
            method_args_iter = get_features_iterable(inp_str, target)
            for method_name, args, kwargs in method_args_iter:
                for ref_obj in self.refined_obj.values():
                    ref_obj.__getattribute__(method_name)(*args, **kwargs)

    def get_common_feature(
        self,
        attr_name,
        attr_args,
        attr_kwargs={},
        as_dataframe=False,
    ):
        """
        Return time series of a specific feature for all symbols in the
        set. The set of features supported are methods of `RefinedData`

        Parameters
        ----------
        `attr_name` : ``str``
            `RefinedData` method or method suffix without the get prefix
        `attr_args` : ``int`` or ``tuple``
            The first one or few positional argumets required in the
            method. All those non-optional before `start` and `stop`
        `attr_kwargs` : ``dict``
            optional arguments given as ``dict``
        `as_dataframe` : ``bool``
            Define the return data structure
            `False` : ``dict {str : pandas.Series }`` (default)
            `True` : ``pandas.DataFrame``

        Return
        ------
        either dictionary or data-frame depending on `as_dataframe`
        ``dict{str : pandas.Series}``
            symbols as keys
        ``pandas.DataFrame``
            symbols as column names
        """
        attr_name = attr_name.split("_")[-1]
        assert_feature(attr_name)
        attr_name = "get_" + attr_name
        feat_dict = {}
        for symbol, ref_obj in self.refined_obj.items():
            feat_dict[symbol] = ref_obj.__getattribute__(attr_name)(
                *attr_args, **attr_kwargs
            )
        if not as_dataframe:
            return feat_dict
        return pd.DataFrame(feat_dict)

    def correlation_matrix_period(
        self, start=None, stop=None, step=1, target="time:close"
    ):
        """
        Correlation among all symbols in the set using a time period

        Parameters
        ----------
        `start` : ``pandas.Timestamp``
            restrict correlation period starting datetime
        `stop` : ``pandas.Timestamp``
            restrict correlation period final datetime
        `step` : ``int`` or ``str``
            the bar step size according to `target` parameter
        `target` : ``str``
            string encoding the bar type and market data to use
            It must be fromatted as "bar_type:data_field" with:
            `bar_type` in ["time", "tick", "volume", "money"]
            `data_field` in ["open", "high", "low", "close", "volume"]

        Return
        ------
        ``pandas.DataFrame``
        """
        assert_target(target)
        bar_type = target.split(":")[0]
        data_method = "get_" + target.split(":")[1]
        nsymbols = len(self.refined_obj.keys())
        mat = np.empty([nsymbols, nsymbols])
        items_pkg = self.refined_obj.items()
        for i, (symbol1, ref_obj1) in enumerate(items_pkg):
            mat[i, i] = 1.0
            for j, (symbol2, ref_obj2) in enumerate(items_pkg):
                if j <= i:
                    continue
                valid_start = start or max(
                    self.symbol_period[symbol1][0],
                    self.symbol_period[symbol2][0],
                )
                valid_stop = stop or min(
                    self.symbol_period[symbol1][1],
                    self.symbol_period[symbol2][1],
                )
                if valid_stop <= valid_start:
                    mat[i, j] = np.nan
                    mat[j, i] = np.nan
                    continue
                kwargs = {
                    "start": valid_start,
                    "stop": valid_stop,
                    "step": step,
                    "bar_type": bar_type,
                }
                data_ser1 = ref_obj1.__getattribute__(data_method)(**kwargs)
                data_ser2 = ref_obj2.__getattribute__(data_method)(**kwargs)
                corr = data_ser1.corr(data_ser2)
                mat[i, j] = corr
                mat[j, i] = corr
        corr_df = pd.DataFrame(
            mat,
            columns=list(self.refined_obj.keys()),
            index=list(self.refined_obj.keys()),
            dtype=np.float64,
        )
        corr_df.name = "CorrelationMatrix"
        corr_df.index.name = "symbols"
        return corr_df

    def moving_correlation(
        self,
        symbol1,
        symbol2,
        window,
        start=None,
        stop=None,
        step=1,
        target="time:close",
    ):
        """
        Compute correlation of two symbols in a moving `window`

        Parameters
        ----------
        `start` : ``pandas.Timestamp``
            restrict correlation period starting datetime
        `stop` : ``pandas.Timestamp``
            restrict correlation period final datetime
        `step` : ``int`` or ``str``
            the bar step size according to `target` parameter
        `target` : ``str``
            string encoding the bar type and market data to use
            It must be fromatted as "bar_type:data_field" with:
            `bar_type` in ["time", "tick", "volume", "money"]
            `data_field` in ["open", "high", "low", "close", "volume"]

        Return
        ------
        ``pandas.Series``
            Series with correlation in the moving window

        """
        assert_target(target)
        bar_type = target.split(":")[0]
        data_method = "get_" + target.split(":")[1]
        valid_start = start or max(
            self.symbol_period[symbol1][0],
            self.symbol_period[symbol2][0],
        )
        valid_stop = stop or min(
            self.symbol_period[symbol1][1],
            self.symbol_period[symbol2][1],
        )
        ref_obj1 = self.refined_obj[symbol1]
        ref_obj2 = self.refined_obj[symbol2]
        kwargs = {
            "start": valid_start,
            "stop": valid_stop,
            "step": step,
            "bar_type": bar_type,
        }
        data_ser1 = ref_obj1.__getattribute__(data_method)(**kwargs)
        data_ser2 = ref_obj2.__getattribute__(data_method)(**kwargs)
        clean_data = pd.concat([data_ser1, data_ser2], axis=1).dropna()
        indexes = clean_data.index
        data_set1 = clean_data.values[:, 0]
        data_set2 = clean_data.values[:, 1]
        mov_corr = np.empty(clean_data.shape[0], dtype=np.float64)
        numba_stats.moving_correlation(
            window,
            data_set1,
            data_set2,
            mov_corr,
        )
        return pd.Series(mov_corr[window - 1 :], indexes[window - 1 :])


@total_ordering
class StockTrade:
    """
    Object to represent stock market shares negociations
    with ordering methods based on datetime
    """

    def __init__(
        self,
        symbol,
        quantity,
        unit_price,
        date,
        flag=1,
        fixed_tax=0,
        relative_tax=0,
        daily_tax=0,
    ):
        """
        Construct a trade order in stock market using all required information

        Parameters
        ----------
        `symbol` : ``str``
            share symbol in stock market
        `quantity` : ``int``
            number of shares negociated
        `unit_price` : ``float``
            unit acquisition price
        `date` : ``pandas.Timestamp``
            date and time the trade occurred
        `flag` : ``int``
            either +1 for buy position and -1 for sell position
        `fixed_tax` : ``float``
            constant value in currency to execute an order in the stock market
        `relative_tax` : ``float``
            fraction of the total order value charged
        `daily_tax` : ``float`` (default 0)
            fraction of share price charged to hold position per day
            Usually only applicable to maintain short/sell position
        """
        self.symbol = symbol
        self.quantity = int(quantity)
        self.unit_price = unit_price
        self.date = date
        self.flag = flag
        self.fixed_tax = fixed_tax
        self.relative_tax = relative_tax
        self.daily_tax = daily_tax
        self.__assert_input_params()

    def __assert_input_params(self):
        if self.quantity < 0:
            raise ValueError(
                "Quantity in a trade must be positive. {} given".format(
                    self.quantity
                )
            )
        if self.unit_price < 0:
            raise ValueError(
                "The stock price cannot be negative. {} given".format(
                    self.unit_price
                )
            )
        if abs(self.flag) != 1 and not isinstance(self.flag, int):
            raise ValueError(
                "Flag in a trade must be 1>buy or -1>sell. {} given".format(
                    self.flag
                )
            )
        if not isinstance(self.date, pd.Timestamp):
            raise ValueError(
                "{} is invalid trade date. Must be pandas.Timestamp".format(
                    self.date
                )
            )
        if self.fixed_tax < 0 or self.relative_tax < 0 or self.daily_tax < 0:
            raise ValueError("All taxes must be positive")

    def __valid_input_date(self, date):
        """Confirm `date` is ahead of `self.date`"""
        if not isinstance(date, pd.Timestamp):
            return False
        return date >= self.date

    def __valid_input_quantity(self, quant):
        """
        Confirm `quant` is integer and smaller than trade `self.quantity`
        """
        if not isinstance(quant, int):
            return False
        return quant <= self.quantity

    def __raise_date_error(self, date):
        raise ValueError(
            "{} is invalid date for trade occurred in {}".format(
                date, self.date
            )
        )

    def __raise_quantity_error(self, quantity):
        raise ValueError(
            "{} is invalid quantity for trade of {} shares".format(
                quantity, self.quantity
            )
        )

    def _valid_comparison(self, other):
        return hasattr(other, "date") and hasattr(other, "symbol")

    def __eq__(self, other):
        if not self._valid_comparison(other):
            return NotImplemented
        return (self.date, self.symbol) == (other.date, other.symbol)

    def __lt__(self, other):
        if not self._valid_comparison(other):
            return NotImplemented
        return (self.date, self.symbol) < (other.date, other.symbol)

    def total_rolling_tax(self, date, quantity=None):
        """Compute the rolling cost of the operation up to `date`"""
        if quantity is None:
            quantity = self.quantity
        if not self.__valid_input_date(date):
            self.__raise_date_error(date)
        if not self.__valid_input_quantity(quantity):
            self.__raise_quantity_error(quantity)
        days_elapsed = 1 + (date - self.date).days
        total_cost = quantity * self.unit_price
        return total_cost * days_elapsed * self.daily_tax

    def rolling_tax(self, date_array, quantity=None):
        """Compute the rolling cost of the operation from time array"""
        if quantity is None:
            quantity = self.quantity
        if not self.__valid_input_quantity(quantity):
            self.__raise_quantity_error(quantity)
        valid_init = date_array.get_loc(self.date, method="backfill")
        dt_arr = date_array[valid_init:]
        days_elapsed_arr = 1 + (dt_arr - self.date).days
        total_cost = quantity * self.unit_price
        tax_arr = (total_cost * days_elapsed_arr * self.daily_tax).values
        return pd.Series(tax_arr, index=dt_arr)

    def total_taxes(self, date=None, quantity=None):
        """
        Return total amount spent with taxes up to `date`
        For a partial quantity (not closing the position)
        do not include the initial fixed tax

        Parameters
        ---
        `date` : ``pandas.Timestamp``
            date to consider in rolling taxes (shares rent if applicable)
        `quantity` : ``int``
            number of shares to compute the taxes

        """
        if quantity is None:
            quantity = self.quantity
        if not self.__valid_input_quantity(quantity):
            self.__raise_quantity_error(quantity)
        inc_fixed = 0  # discard fixed tax if not close position
        if quantity == self.quantity:
            inc_fixed = self.fixed_tax
        if self.flag > 0:
            return inc_fixed + self.relative_tax * quantity * self.unit_price
        return (
            inc_fixed
            + self.relative_tax * quantity * self.unit_price
            + self.total_rolling_tax(date, quantity)
        )

    def raw_invest(self, current_price=None, quantity=None):
        """
        Compute the value invested in the trade up to `date`
        Note that for short position it requires the price of
        the shares, since the investiment needs current price

        Parameters
        ---
        `current_price` : ``float``
            Only required for sell/short position
        `quantity` : ``int``
            number of shares to consider. Default `self.quantity`

        return
        ---
        ``float``
            value in currency invested according to `quantity`

        """
        if quantity is None:
            quantity = self.quantity
        if not self.__valid_input_quantity(quantity):
            self.__raise_quantity_error(quantity)
        if self.flag > 0:
            return quantity * self.unit_price
        if current_price is None:
            raise ValueError("Need current price to compute short investment")
        return quantity * current_price

    def total_invest(self, date=None, current_price=None, quantity=None):
        """
        Return the total amount in local currency required including taxes
        """
        return self.raw_invest(current_price, quantity) + self.total_taxes(
            date, quantity
        )

    def raw_result(self, unit_price, quantity=None):
        """Return the raw result ignoring taxes for `quantity` shares"""
        if quantity is None:
            quantity = self.quantity
        if not self.__valid_input_quantity(quantity):
            self.__raise_quantity_error(quantity)
        return quantity * (unit_price - self.unit_price) * self.flag

    def net_result(self, unit_price, date=None):
        """
        Return the net result of the operation up to `date` after taxes

        Parameters
        ---
        `unit_price` : ``float``
            share unit price in current date
        `date` : ``pandas.Timestamp``
            date is only required to compute rolling tax of short operation

        Return
        ---
        ``float``
            result in local currency of the operation
        """
        if not self.__valid_input_date(date):
            self.__raise_date_error(date)
        taxes = self.total_taxes(date, self.quantity)
        return self.raw_result(unit_price, self.quantity) - taxes

    def raw_result_series(self, unit_price_series, quantity=None):
        """Equivalent to ``self.net_result_series`` but ignoring taxes"""
        if quantity is None:
            quantity = self.quantity
        valid_init = unit_price_series.index.get_loc(
            self.date, method="backfill"
        )
        valid_prices = unit_price_series[valid_init:]
        return quantity * (valid_prices - self.unit_price) * self.flag

    def net_result_series(self, unit_price_series, quantity=None):
        """
        Compute total return time series in local currency discounting taxes
        If `quantity < self.quantity` corresponding to a partial close, then
        the fixed tax is ignored. This is a suitable behavior in portfolio
        returns computation when partial closes occur and avoid counting the
        fixed tax twice unmounting the same position

        Parameters
        ----------
        `unit_price_series` : ``pandas.Series``
            Series with time indexing corresponding to stock prices
            The index final time must be larger than `self.date`
        `quantity` : ``int``
            number of shares to consider. Default is `self.quantity`

        Return
        ------
        ``pandas.Series``
        """
        if quantity is None:
            quantity = self.quantity
        roll_tax = self.rolling_tax(unit_price_series.index, quantity)
        rela_tax = quantity * self.unit_price * self.relative_tax
        if quantity == self.quantity:
            total_tax = self.fixed_tax + rela_tax + roll_tax
        else:
            total_tax = rela_tax + roll_tax
        return self.raw_result_series(unit_price_series, quantity) - total_tax


class PortfolioRecord:
    """
    Class to record a set of trades. Basically it holds as attribute a
    list of `StockDeal` objects ordered by date they ocurred. As date
    object all methods use `pandas.Timestamp`
    """

    def __init__(self):
        """Initialize empty list of trades"""
        self.__trades = []

    def empty_history(self):
        """Return True if there are no trades registered"""
        return not self.__trades

    def get_trades_dataframe(self, date=None, symbol=None):
        """
        Return trades information as dataframe

        Parameters
        ----------
        `date` : ``pandas.Timestamp``
            Maximum date to consider. Discard all trades after it
        `symbol` : ``str``
            filter for a unique symbol

        Return
        ------
        ``pandas.DataFrame``
        """
        if not date:
            trades = self.get_all_trades()
        else:
            trades = self.get_trades_before(date)
        df_list = []
        invest_dict = {}
        for trade in trades:
            if trade.symbol not in invest_dict.keys():
                invest_dict[trade.symbol] = (
                    trade.flag * trade.quantity,
                    trade.unit_price,
                )
            else:
                old_quant = invest_dict[trade.symbol][0]
                old_price = invest_dict[trade.symbol][1]
                new_quant = old_quant + trade.flag * trade.quantity
                if trade.flag * old_quant > 0:
                    new_price = (
                        old_price * old_quant
                        + trade.flag * trade.quantity * trade.unit_price
                    ) / new_quant
                else:
                    if trade.quantity > invest_dict[trade.symbol][0]:
                        new_price = trade.unit_price
                    else:
                        new_price = old_price
                invest_dict[trade.symbol] = (new_quant, new_price)
            if trade.flag > 0:
                flag_info = "buy"
            else:
                flag_info = "sell"
            total_quant = invest_dict[trade.symbol][0]
            mean_price = invest_dict[trade.symbol][1]
            df_row = [
                trade.date,
                trade.symbol,
                trade.quantity,
                trade.unit_price,
                flag_info,
                total_quant,
                mean_price,
            ]
            df_list.append(df_row)
        raw_df = pd.DataFrame(
            df_list,
            columns=[
                "DateTime",
                "Symbol",
                "Quantity",
                "Price",
                "FlagInfo",
                "TotalQuantity",
                "MeanPrice",
            ],
        )
        if symbol is not None:
            return raw_df.loc[raw_df.Symbol == symbol].set_index("DateTime")
        return raw_df.set_index("DateTime")

    def get_all_trades(self):
        """Return the list of all trades recorded"""
        return self.__copy_trades(self.__trades)

    def get_trades_symbol(self, symbol):
        """Return list with all trades involving `symbol`"""
        sym_trades = []
        for trade in self.__trades:
            if trade.symbol == symbol:
                sym_trades.append(trade)
        return self.__copy_trades(sym_trades)

    def get_trades_before(self, date):
        """Return list of trades objects occurred before `date`"""
        i = 0
        if not date:
            return self.get_all_trades()
        for trade in self.__trades:
            if trade.date >= date:
                break
            i = i + 1
        return self.__copy_trades(self.__trades[:i])

    def get_trades_after(self, date):
        """Return list of trades objects occurred afeter `date`"""
        i = 0
        if not date:
            return self.get_all_trades()
        for trade in self.__trades:
            if trade.date > date:
                break
            i = i + 1
        return self.__copy_trades(self.__trades[i:])

    def delete_trades_symbol(self, symbol):
        """Remove from records all trades related to the `symbol`"""
        trades_to_remove = self.get_trades_symbol(symbol)
        for trade in trades_to_remove:
            self.__trades.remove(trade)

    def delete_trades_before(self, date):
        """Delete all trades that occurred before `date`"""
        i = 0
        for trade in self.__trades:
            if trade.date >= date:
                break
            i = i + 1
        self.__trades = self.__trades[i:]

    def delete_trades_after(self, date):
        """Delete all trades that occurred after `date`"""
        i = 0
        for trade in self.__trades:
            if trade.date > date:
                break
            i = i + 1
        self.__trades = self.__trades[:i]

    def record_trade(
        self,
        symbol,
        quantity,
        unit_price,
        date,
        flag=1,
        fixed_tax=0,
        relative_tax=0,
        daily_tax=0,
    ):
        """
        Insert new trade in the records

        Parameters
        ---
        `symbol` : ``str``
            symbol code in stock market
        `quantity` : ``int``
            number of shares negotiated
        `unit_price` : ``float``
            price of the acquisition
        `date` : ``pandas.Timestamp``
            Time instant the trade happened
        `flag` : ``int`` either +1 or -1
            +1 for buy operation and -1 for sell operation
        `fixed_tax` : ``float``
            contant value in currency to execute an order in the stock market
        `relative_tax` : ``float``
            fraction of the total order value
        `daily_tax` : ``float``
            possibly tax to hold the position each day.
            Useful to describe short positions. Default 0 for long positions

        """
        bisect.insort(
            self.__trades,
            StockTrade(
                symbol,
                quantity,
                unit_price,
                date,
                flag,
                fixed_tax,
                relative_tax,
                daily_tax,
            ),
        )

    def __copy_trades(self, list_of_trades):
        copy_list = []
        for trade in list_of_trades:
            bisect.insort(
                copy_list,
                StockTrade(
                    trade.symbol,
                    trade.quantity,
                    trade.unit_price,
                    trade.date,
                    trade.flag,
                    trade.fixed_tax,
                    trade.relative_tax,
                    trade.daily_tax,
                ),
            )
        return copy_list

    def __str__(self):
        return str(self.get_trades_dataframe())


class Portfolio(PortfolioRecord, RefinedSet):
    """
    This class provide methods to dynamically simulate a portfolio
    with some real aspects, including taxes and the possibility of
    set sell orders without suficient shares (short position)

    This class also aims to provide a simple interface to analyze
    results from multiple shares and compute the results (profit/
    loss) at any input date-time. In this way its design is two-fold
    aimed, first to use alongside orders requested from other models
    and track the results of these orders, and second to serve as
    input in a trainning model

    Inherit
    ---
    ``PortfolioRecord``
        Class to maintain records of all orders
    ``RefinedSet``
        Class to provide statistical feature for set of shares

    """

    def __init__(
        self,
        db_path,
        fixed_tax=0,
        relative_tax=0,
        preload={},
        common_features={},
    ):
        """
        Define a portfolio subject to some operational taxes
        and commom features to be analyzed for all companies

        Parameters
        ---
        `db_path` : ``str``
            full path (with filename) to database file
        `fixed_tax` : ``float``
            fixed tax charged independent of the volume
        `relative_tax` : ``float``
            relative fraction of the operation price charged in an order
            Usually very small (tipically < 0.0005)
        `preload` : ``dict {str : list}``
            dictionary needed in `RefinedSet` constructor
        `commom_features` : ``dict { str : str }``
            dictionaty needed in `RefinedSet` constructor

        """
        ref_set_args = (db_path, preload, common_features)
        self.fixed_tax = fixed_tax
        self.relative_tax = relative_tax
        RefinedSet.__init__(self, *(ref_set_args))
        PortfolioRecord.__init__(self)

    def __nearest_index(self, symbol, date):
        """Return the nearest valid index of the dataframe"""
        return self.refined_obj[symbol].df.index.get_loc(
            date, method="backfill"
        )

    def __nearest_date(self, symbol, date):
        """Return the nearest valid date in dataframe"""
        return self.refined_obj[symbol].df.index[
            self.__nearest_index(symbol, date)
        ]

    def __approx_price(self, symbol, date):
        """Return approximate price (using nearest available date)"""
        return self.refined_obj[symbol].df.Close[
            self.__nearest_index(symbol, date)
        ]

    def set_position_buy(
        self, symbol, date, quantity, sl=None, tp=None, ih=None
    ):
        """
        Insert a buy(long) position in the portfolio with possible halt
        if at least one of `sl`(stop loss), `tp`(take profit) or `ih`
        (invetment horizon) is given and the underlying condition is
        fulfilled, automatically set a sell order to close the trade

        Parameters
        ---
        `symbol` : ``str``
            share symbol in the stock market
        `date` : ``pandas.Timestamp``
            date the order was executed
        `quantity` : ``int`` (positive)
            How many shares to buy
        `sl` : ``float`` between (0, 1)
            relative loss tolerance of the investment (stop loss)
            1.0 is the maximum loss = 100%
        `tp` : ``float`` > 0
            relative profit tolerance of the investment (take profit)
        `ih` : ``int`` or ``pandas.Timedelta``
            time tolerance to close the trade. Case integer
            indicate the number of consecutive bars to wait

        """
        if tp is not None and tp <= 0:
            raise ValueError("Invalid take profit {} for buy order".format(tp))
        if sl is not None:
            if sl <= 0 or sl >= 100.0:
                raise ValueError(
                    "Invalid stop loss {} for buy order".format(sl)
                )
        self.new_refined_symbol(symbol)
        valid_date = self.__nearest_date(symbol, date)
        approx_price = self.__approx_price(symbol, date)
        self.record_trade(
            symbol,
            quantity,
            approx_price,
            valid_date,
            1,
            self.fixed_tax,
            self.relative_tax,
        )
        # If barriers are set, automatically include the sell operation
        cls_date, result = event_label(
            1, self.refined_obj[symbol].df.loc[valid_date:], sl, tp, ih
        )
        if ih is None and result == 0:
            return
        self.set_position_sell(symbol, cls_date, quantity)

    def set_position_sell(
        self,
        symbol,
        date,
        quantity,
        sl=None,
        tp=None,
        ih=None,
        daily_tax=0.0,
    ):
        """
        Register a sell order(short) in the portfolio

        Parameters
        ---
        `symbol` : ``str``
            share symbol in the stock market
        `date` : ``pandas.Timestamp``
            date the order was executed
        `quantity` : ``int`` (positive)
            How many shares to sell
        `sl` : ``float``
            stop loss of the positioning in percent value (4 means 4%)
        `tp` : ``float``
            take profit to close the positioning in percent value (max 100)
        `ih` : ``int`` or ``pandas.Timedelta``
            time tolerance to close the trade. Case integer
            indicate the number of consecutive bars to wait
        `dialy_tax` : ``float``
            the rent fraction required to maintain the order per day

        """
        if tp is not None:
            if tp >= 100.0 or tp <= 0:
                raise ValueError(
                    "Invalid take profit {} for sell order".format(tp)
                )
        if sl is not None and sl <= 0:
            raise ValueError("Invalid stop loss {} for sell order".format(sl))
        self.new_refined_symbol(symbol)
        valid_date = self.__nearest_date(symbol, date)
        approx_price = self.__approx_price(symbol, date)
        self.record_trade(
            symbol,
            quantity,
            approx_price,
            valid_date,
            -1,
            self.fixed_tax,
            self.relative_tax,
            daily_tax,
        )
        # If barriers are set, automatically include the buy position
        cls_date, result = event_label(
            -1, self.refined_obj[symbol].df.loc[valid_date:], sl, tp, ih
        )
        if ih is None and result == 0:
            return
        self.set_position_buy(symbol, cls_date, quantity)

    def set_from_labels(self, symbol, nshares, labels_df):
        """
        Integrate with labels module to set operations from a strategy

        Parameters
        ----------
        `symbol` : ``str``
            company symbol the trades refers to
        `nshares` : ``int`` or ``iterable(int)``
            number of shares for each trade labeled from strategy
        `labels_df` : ``pandas.DataFrame``
            output from ``mvp.labels.event_label_series`` see the
            function doc and eventually ``mvp.primary`` if needed
            to generate strategy triggers
        """
        if isinstance(nshares, int):
            nshares = nshares * np.ones(labels_df.index.size, dtype=int)
        for init, side, end, quantity in zip(
            labels_df.index, labels_df.Side, labels_df.PositionEnd, nshares
        ):
            if side > 0:
                self.set_position_buy(symbol, init, quantity)
                self.set_position_sell(symbol, end, quantity)
            else:
                self.set_position_sell(symbol, init, quantity)
                self.set_position_buy(symbol, end, quantity)

    def overall_net_result(self, as_dataframe=False):
        """
        Compute ``self.symbol_net_result`` for all present symbols
        `as_dataframe` defines the format of return datatype, thus
        the result series of all symbols are constrained to latest
        date among all indexes

        Return
        ------
        ``dict { str : pandas.Series }``
            key values pairs as `symbol:returns`
        or
        ``pandas.DataFrame``
            symbols labeling the columns
        """
        res_dict = {}
        for symbol in self.refined_obj.keys():
            res_dict[symbol] = self.symbol_net_result(symbol, True)
        if not as_dataframe:
            return res_dict
        max_date = min([series.index[-1] for series in res_dict.values()])
        df = pd.DataFrame(res_dict).fillna(0)
        df["Total"] = df.sum(axis=1)
        return df.loc[:max_date]

    def symbol_net_result(self, symbol, cummulative=True):
        """
        Compute the total return series for selected `symbol`
        in local currency. The `cummulative` hold the result
        of all trades along portfolio history to provide the
        all time returns. For example, if the last trade was
        in some time instant `t`, after that the series will
        display a constant value corresponding to the result
        at the last trade point
        """
        if not self.has_symbol(symbol):
            return pd.Series([], dtype=float)
        trades = self.get_trades_symbol(symbol)[::-1]
        first_trade_date = trades[-1].date
        last_trade_date = trades[0].date
        prices = self.refined_obj[symbol].get_close(first_trade_date)
        full_ret = pd.Series(np.zeros(prices.size), index=prices.index)
        trade_stack = [trades.pop()]  # stack of successive equal side trades
        ret_list = []  # record return series every time a trade is closed
        while trades:
            trade_stack_flag = trade_stack[0].flag
            while trades and trades[-1].flag == trade_stack_flag:
                trade_stack.append(trades.pop())
            if not trades:
                break
            reverse_trade = trades.pop()
            quant = reverse_trade.quantity
            dt = reverse_trade.date
            while trade_stack and quant >= trade_stack[-1].quantity:
                pop_stack = trade_stack.pop()
                ret_list.append(pop_stack.net_result_series(prices[:dt]))
                quant = quant - pop_stack.quantity
            if trade_stack:
                # partial close
                if quant > 0:
                    ret_list.append(
                        trade_stack[-1].net_result_series(prices[:dt], quant)
                    )
                    trade_stack[-1].quantity -= quant
                    quant = 0
            elif quant > 0:
                # invert position since stack is empty
                new_stack_init = StockTrade(
                    reverse_trade.symbol,
                    quant,
                    reverse_trade.unit_price,
                    reverse_trade.date,
                    reverse_trade.flag,
                    reverse_trade.fixed_tax,
                    reverse_trade.relative_tax,
                    reverse_trade.daily_tax,
                )
                trade_stack.append(new_stack_init)
            elif trades:
                # reverse trade exactly closed all open positions(empty stack)
                trade_stack.append(trades.pop())
        while ret_list:
            # compute returns of closed trades
            trade_ret = ret_list.pop()
            if cummulative:
                end_dt = trade_ret.index[-1]
                i = prices.index.get_loc(end_dt, method="backfill") + 1
                extend_ind = prices.index[i:]
                after_ret = pd.Series(trade_ret[-1], index=extend_ind)
                trade_ret = pd.concat([trade_ret, after_ret])
            fill_copy = full_ret.copy()
            full_ret = (full_ret + trade_ret).fillna(fill_copy)
        for trade in trade_stack:
            # compute returns of opened positions that remains on stack
            trade_ret = trade.net_result_series(prices)
            fill_copy = full_ret.copy()
            full_ret = (full_ret + trade_ret).fillna(fill_copy)
        if trade_stack or cummulative:
            # if stack is not empty there are open positions
            last_trade_date = prices.size
        return full_ret[:last_trade_date]
