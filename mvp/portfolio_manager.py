""" Portfolio module

This module provide statistical features of a set of companies
and implement basic tools to obtain portfolio performance

"""

import os
import bisect
import numpy as np
import pandas as pd

from functools import total_ordering
from mvp.refined_data import RefinedData, assert_target, assert_feature
from mvp.utils import numba_stats
from mvp.utils import validate_features_string, get_features_iterable


class RefinedSet:
    """
    Class to collect a set of stock shares and obtain common features
    For more info about the features see `RefinedData` class. Despite
    the possibility to compute new features acessing refined objects
    by the `self.refined_obj` dictionary, for better performance, the
    useage of `common_features` is recomended. These `common_features`
    does not change the workflow to access some feature, though they
    can improve performance since the requested features are kept in
    cache memory. Moreover new symbols introduced automatic set these
    features in cache memory as well.

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
            Available keys are methods of `RawData` class which
            end with the suffix "_bars" and the value is `step`
            argument required in these methods. Some examples
            {
                "time" : ``int`` or "day"
                "tick" : ``int``
                "volume" : ``int``
                "money" : ``int``
            }
            The value field also support ``list`` of these types
        `cache_common_features` : ``dict {str : str}``
            dictionary to preload set of common features for all symbols
            introduced in this `RefinedSet` through `RefinedData` object
            KEYS:
                The keys must be formated as `"bar_type:data_field"`
                where `bar_type` inform how data bars are formed and
                `data_field` the bar value to use. Some examples are
                    "time:close"  - use close price in time-spaced bars
                    "time:volume" - use volume traded in time-spaced bars
                    "tick:high"   - use high price in tick-spaced bars
                    "money:close" - use close price in money-spaced bars
                This disctionary keys is also referred to as `target`
                which is an argument of `RefinedData` methods.
                The available names for `bar_type` are the same of
                the keys of `preload` parameter and are the methods
                of `RawData` that has as suffix `_bars`
                The available names for `data_field` are suffixes of
                any `RawData` method that starts with `get_`
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
        """ For each refined object in this set clean the cache """
        for ref_obj in self.refined_obj.values():
            ref_obj.cache_clean()

    def is_empty(self):
        """
        Return `True` if there are no symbols refined in this object
        """
        return not self.refined_obj

    def has_symbol(self, symbol):
        """ Return True if the `symbol` (or list of) is in the set """
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
        Print on screen current status of this symbol set object
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
            valid symbol contained in the `self.db_path` database
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
        Return time series of a specific feature for all symbols in the set
        The set of features supported are methods of `RefinedData`

        Parameters
        ----------
        `attr_name` : ``str``
            `RefinedData` method or method suffix without the get prefix
        `attr_args` : ``int`` or ``tuple``
            The first one or few argumets required in method
            all those non-optional before `start` and `stop`
        `attr_kwargs` : ``dict``
            optional arguments given as ``dict``
        `as_dataframe` : ``bool``
            Define the return data structure
            `False` : ``dict`` (default)
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

        Return
        ---
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
class StockDeal:
    """
    Object to represent stock market shares negociations with ordering
    methods based on date and time.

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
        Construct a deal order in stock market using all required information

        Parameters
        ---
        `symbol` : ``str``
            share symbol in stock market
        `quantity` : ``int``
            number of shares negociated
        `unit_price` : ``float``
            unit acquisition price
        `date` : ``pandas.Timestamp``
            date and time the deal occurred
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
        self.deal_date = date
        self.flag = flag
        self.fixed_tax = fixed_tax
        self.relative_tax = relative_tax
        self.daily_tax = daily_tax
        self.__assert_input_params()

    def __assert_input_params(self):
        if self.quantity < 0:
            raise ValueError(
                "Quantity in a deal must be positive. {} given".format(
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
                "Flag in a deal must be 1>buy or -1>sell. {} given".format(
                    self.flag
                )
            )
        if not isinstance(self.deal_date, pd.Timestamp):
            raise ValueError(
                "{} is invalid deal date. Must be pandas.Timestamp".format(
                    self.deal_date
                )
            )
        if self.fixed_tax < 0 or self.relative_tax < 0 or self.daily_tax < 0:
            raise ValueError("All taxes must be positive")

    def __valid_input_date(self, date):
        """ Confirm `date` is ahead of `self.deal_date` """
        if not isinstance(date, pd.Timestamp):
            return False
        return date >= self.deal_date

    def __valid_input_quantity(self, quant):
        """
        Confirm `quant` is integer and smaller than deal `self.quantity`
        """
        if not isinstance(quant, int):
            return False
        return quant <= self.quantity

    def __raise_date_error(self, date):
        raise ValueError(
            "{} is invalid date for deal occurred in {}".format(
                date, self.deal_date
            )
        )

    def __raise_quantity_error(self, quantity):
        raise ValueError(
            "{} is invalid quantity for deal of {} shares".format(
                quantity, self.quantity
            )
        )

    def _valid_comparison(self, other):
        return hasattr(other, "deal_date") and hasattr(other, "symbol")

    def __eq__(self, other):
        if not self._valid_comparison(other):
            return NotImplemented
        return (self.deal_date, self.symbol) == (other.deal_date, other.symbol)

    def __lt__(self, other):
        if not self._valid_comparison(other):
            return NotImplemented
        return (self.deal_date, self.symbol) < (other.deal_date, other.symbol)

    def total_rolling_tax(self, date, quantity=None):
        """ Compute the rolling cost of the operation up to `date` """
        if quantity is None:
            quantity = self.quantity
        if not self.__valid_input_date(date):
            self.__raise_date_error(date)
        if not self.__valid_input_quantity(quantity):
            self.__raise_quantity_error(quantity)
        days_elapsed = 1 + (date - self.deal_date).days
        total_cost = quantity * self.unit_price
        return total_cost * days_elapsed * self.daily_tax

    def time_rolling_tax(self, date_array, quantity=None):
        """ Compute the rolling cost of the operation from time array """
        if quantity is None:
            quantity = self.quantity
        if not self.__valid_input_quantity(quantity):
            self.__raise_quantity_error(quantity)
        days_elapsed_arr = 1 + (date_array - self.deal_date).days
        total_cost = quantity * self.unit_price
        tax_arr = (total_cost * days_elapsed_arr * self.daily_tax).values
        return pd.Series(tax_arr, index=date_array)

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
        Compute the value invested in the deal up to `date`
        Note that for short position it require the price of
        the shares since the investiment is aways a buy order

        Parameters
        ---
        `current_price` : ``float``
            Only required for sell deal
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
        Return the total amount in currency required including taxes
        See also the `self.raw_invest` method which ignore taxes
        """
        return self.raw_invest(current_price, quantity) + self.total_taxes(
            date, quantity
        )

    def raw_result(self, unit_price, quantity=None):
        """ Return the raw result discounting the taxes up to `date` """
        if quantity is None:
            quantity = self.quantity
        if not self.__valid_input_quantity(quantity):
            self.__raise_quantity_error(quantity)
        return quantity * (unit_price - self.unit_price) * self.flag

    def net_result(self, unit_price, date=None):
        """
        Return the net result of the operation up to `date`

        Parameters
        ---
        `date` : ``pandas.Timestamp``
            date to consider in net profit/loss computation
        `unit_price` : ``float``
            share unit price in current date

        Return
        ---
        ``float``
            result in currency of the operation in case it is closed

        See also `partial_close`

        """
        if not self.__valid_input_date(date):
            self.__raise_date_error(date)
        taxes = self.total_taxes(date, self.quantity)
        return self.raw_result(unit_price, self.quantity) - taxes

    def partial_close(self, unit_price, cls_quant, date=None):
        """
        Return partial result due to reduction in position
        `self.quantity` is reducing it by `cls_quant`

        Parameters
        ---
        `date` : ``pandas.Timestamp``
            date the order ocurred
        `unit_price` : ``float``
            share price in the current `date`
        `cls_quant` : ``int``
            how many shares were sold/buyed. Must be smaller than self.quantity

        Return
        ---
        ``float``
            result in currency of the operation

        Modified
        ---
        `self.quantity`
            reduced by `cls_quant`

        """
        if not self.__valid_input_quantity(cls_quant):
            self.__raise_quantity_error(cls_quant)
        taxes = self.total_taxes(date, cls_quant)
        op_result = self.raw_result(unit_price, cls_quant) - taxes
        self.quantity -= cls_quant
        return op_result


class PortfolioRecord:
    """
    Class to record a set of deals. Basically it holds as attribute a
    list of `StockDeal` objects ordered by date they ocurred. As date
    object all methods use `pandas.Timestamp`

    """

    def __init__(self):
        """ Initialize empty list of deals """
        self.__deals = []

    def empty_history(self):
        """ Return True if there are no deals registered """
        return not self.__deals

    def get_deals_dataframe(self, date=None):
        """ Return all deals information as dataframe up to `date` """
        if not date:
            deals = self.__deals
        else:
            deals = self.get_deals_before(date)
        df_list = []
        invest_dict = {}
        for deal in deals:
            if deal.symbol not in invest_dict.keys():
                invest_dict[deal.symbol] = (
                    deal.flag * deal.quantity,
                    deal.unit_price,
                )
            else:
                old_quant = invest_dict[deal.symbol][0]
                old_price = invest_dict[deal.symbol][1]
                new_quant = old_quant + deal.flag * deal.quantity
                if deal.flag * old_quant > 0:
                    new_price = (
                        old_price * old_quant
                        + deal.flag * deal.quantity * deal.unit_price
                    ) / new_quant
                else:
                    if deal.quantity > invest_dict[deal.symbol][0]:
                        new_price = deal.unit_price
                    else:
                        new_price = old_price
                invest_dict[deal.symbol] = (new_quant, new_price)
            if deal.flag > 0:
                flag_info = "buy"
            else:
                flag_info = "sell"
            total_quant = invest_dict[deal.symbol][0]
            mean_price = invest_dict[deal.symbol][1]
            df_row = [
                deal.deal_date,
                deal.symbol,
                deal.quantity,
                deal.unit_price,
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
        return raw_df.set_index("DateTime")

    def get_all_deals(self):
        """ Return the list of all deals recorded """
        return self.__deals.copy()

    def get_deals_symbol(self, symbol):
        """ Return list with all deals of `symbol` """
        sym_deals = []
        for deal in self.__deals:
            if deal.symbol == symbol:
                sym_deals.append(deal)
        return sym_deals

    def get_deals_before(self, date):
        """ Return list of deals objects occurred before `date` """
        i = 0
        if not date:
            return self.get_all_deals()
        for deal in self.__deals:
            if deal.deal_date >= date:
                break
            i = i + 1
        return self.__deals[:i].copy()

    def get_deals_after(self, date):
        """ Return list of deals objects occurred afeter `date` """
        i = 0
        if not date:
            return self.get_all_deals()
        for deal in self.__deals:
            if deal.deal_date > date:
                break
            i = i + 1
        return self.__deals[i:].copy()

    def delete_deals_symbol(self, symbol):
        """ Remove from records all deals related to the given `symbol` """
        deals_to_remove = self.get_deals_symbol(symbol)
        for deal in deals_to_remove:
            self.__deals.remove(deal)

    def delete_deals_before(self, date):
        """ Delete all deals that occurred before `date` """
        i = 0
        for deal in self.__deals:
            if deal.deal_date >= date:
                break
            i = i + 1
        self.__deals = self.__deals[i:]

    def delete_deals_after(self, date):
        """ Delete all deals that occurred after `date` """
        i = 0
        for deal in self.__deals:
            if deal.deal_date > date:
                break
            i = i + 1
        self.__deals = self.__deals[:i]

    def record_deal(
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
        Insert new deal in the records

        Parameters
        ---
        `symbol` : ``str``
            symbol code in stock market
        `quantity` : ``int``
            number of shares negotiated
        `unit_price` : ``float``
            price of the acquisition
        `date` : ``pandas.Timestamp``
            Time instant the deal happened
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
            self.__deals,
            StockDeal(
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

    def __str__(self):
        return str(self.get_deals_dataframe())


class Portfolio(PortfolioRecord, RefinedSet):
    """
    Portfolio Class
    ---
    This class provide methods to dynamically simulate a portfolio
    with some real aspects, including taxes and the possibility of
    set sell orders without suficient shares in the portfolio

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
        common_features="",
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
            Usually very small (tipically < 0.001)
        `preload` : ``dict {str : list}``
            dictionary needed in `RefinedSet` constructor
        `commom_features` : ``dict { str : str }``
            dictionaty needed in `RefinedSet` constructor

        """
        ref_set_args = (db_path, common_features, preload)
        self.fixed_tax = fixed_tax
        self.relative_tax = relative_tax
        RefinedSet.__init__(self, *(ref_set_args))
        PortfolioRecord.__init__(self)
        self.portfolio_start = None
        self.portfolio_end = None

    def __nearest_index(self, symbol, date):
        """ Return the nearest valid index of the dataframe """
        return self.refined_obj[symbol].df.index.get_loc(
            date, method="backfill"
        )

    def __nearest_date(self, symbol, date):
        """ Return the nearest valid date in dataframe """
        return self.refined_obj[symbol].df.index[
            self.__nearest_index(symbol, date)
        ]

    def __approx_price(self, symbol, date):
        """ Return approximate price (using nearest available date) """
        return self.refined_obj[symbol].df.Close[
            self.__nearest_index(symbol, date)
        ]

    def __update_portfolio_period(self, date):
        """ Update portfolio period based in new buy or sell event """
        delta = pd.Timedelta(days=1)
        if not self.portfolio_start:
            self.portfolio_start = date - delta
        elif date < self.portfolio_start:
            self.portfolio_start = date - delta
        if not self.portfolio_end:
            self.portfolio_end = date + delta
        elif date > self.portfolio_end:
            self.portfolio_end = date + delta

    def __assert_invest_horizon(self, date_init, ih):
        """ Return boolean Assessing investment horizon from initial date """
        if isinstance(ih, pd.Timedelta) or isinstance(ih, int):
            return True
        if isinstance(ih, pd.Timestamp) and ih >= date_init:
            return True
        raise ValueError(
            "investment horizon date type error: "
            "{} given for {} initial date".format(ih, date_init)
        )

    def __query_stop_date(self, date_init, symbol, sl, tp, ih, flag):
        """
        Find the closest event that happen after `date_init`
        regarded as date of a buy or sell order

        Parameters
        ---
        `date_init` : ``pandas.Timestamp``
            date an order (trade open position) was executed
        `symbol` : ``str``
            valid symbol in the portfolio
        `sl` : ``float`` or None
            Loss tolerance to close the trade in currency
        `tp` : ``float`` or None
            Profit tolerance to close the trade in currency
        `ih` : ``pandas.Timedelta`` or ``pandas.Timestamp`` or ``int`` or None
            investment horizon after opening a position in the market

        Return
        ---
        None
            if none of `sl`, `tp` or `ih` is given
        ``pandas.Timestamp``
            First occurrance of either `sl`, `tp` or `ih`

        """
        if sl is None and tp is None and ih is None:
            return None
        cls_price = self.refined_obj[symbol].df.Close
        if ih is None:
            ih = cls_price.index[-1]
        self.__assert_invest_horizon(date_init, ih)
        if isinstance(ih, pd.Timedelta):
            date_max = date_init + ih
        elif isinstance(ih, int):
            time_step = cls_price.index[1] - cls_price.index[0]
            date_max = date_init + (ih + 1) * time_step
        else:
            date_max = ih + pd.Timedelta(minutes=1)
        hor_price = cls_price.loc[date_init:date_max]
        if sl is not None:
            try:
                if flag > 0:
                    sl_date = hor_price[hor_price <= sl].index[0]
                else:
                    sl_date = hor_price[hor_price >= sl].index[0]
            except IndexError:
                sl_date = hor_price.index[-1]
        else:
            sl_date = hor_price.index[-1]
        if tp is not None:
            try:
                if flag > 0:
                    tp_date = hor_price[hor_price >= tp].index[0]
                else:
                    tp_date = hor_price[hor_price <= tp].index[0]
            except IndexError:
                tp_date = hor_price.index[-1]
        else:
            tp_date = hor_price.index[-1]
        return min(hor_price.index[-1], sl_date, tp_date)

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
        `ih` : ``int`` or ``pandas.Timestamp`` of ``pandas.Timedelta``
            time tolerance to close the trade

        """
        if tp is not None and sl is not None:
            if tp <= 0 or sl >= 1.0 or sl <= 0:
                raise ValueError(
                    "Invalid take profit {} or stop loss {} "
                    "for buy order".format(tp, sl)
                )
        self.new_refined_symbol(symbol)
        valid_date = self.__nearest_date(symbol, date)
        approx_price = self.__approx_price(symbol, date)
        self.record_deal(
            symbol,
            quantity,
            approx_price,
            valid_date,
            1,
            self.fixed_tax,
            self.relative_tax,
        )
        self.__update_portfolio_period(valid_date)
        # convert relative values of stop loss/take profit to prices
        if sl is not None:
            sl = approx_price * (1 - sl)
        if tp is not None:
            tp = approx_price * (1 + tp)
        cls_date = self.__query_stop_date(valid_date, symbol, sl, tp, ih, 1)
        if cls_date is not None:
            self.set_position_sell(symbol, cls_date, quantity)
            self.__update_portfolio_period(cls_date)

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
        `sl` : ``float`` > 0
            relative loss tolerance of the investment (stop loss)
        `tp` : ``float`` between (0, 1)
            relative profit tolerance of the investment (take profit)
            1 is the maximum profit = 100%
        `ih` : ``int`` or ``pandas.Timestamp`` of ``pandas.Timedelta``
            time tolerance to close the trade
        `dialy_tax` : ``float``
            the rent fraction required to maintain the order per day

        """
        if tp is not None and sl is not None:
            if tp >= 1.0 or tp <= 0 or sl <= 0:
                raise ValueError(
                    "Invalid take profit {} or stop loss {} "
                    "for sell order".format(tp, sl)
                )
        self.new_refined_symbol(symbol)
        valid_date = self.__nearest_date(symbol, date)
        approx_price = self.__approx_price(symbol, date)
        self.record_deal(
            symbol,
            quantity,
            approx_price,
            valid_date,
            -1,
            self.fixed_tax,
            self.relative_tax,
            daily_tax,
        )
        self.__update_portfolio_period(valid_date)
        # convert relative values of stop loss/take profit to prices
        if sl is not None:
            sl = approx_price * (1 + sl)
        if tp is not None:
            tp = approx_price * (1 - tp)
        cls_date = self.__query_stop_date(valid_date, symbol, sl, tp, ih, -1)
        if cls_date is not None:
            self.set_position_buy(symbol, cls_date, quantity)
            self.__update_portfolio_period(cls_date)

    def symbol_positioning(self, date):
        """
        Compute position up to `date` for each symbol in terms of quantity
        Negative quantity means a short position

        Return
        ---
        ``dict`` : {``str`` : ``int``}
            share symbol as key and quantity as values in current date

        """
        deals = self.get_deals_before(date)
        pos = {}
        for deal in deals:
            if deal.symbol in pos.keys():
                pos[deal.symbol] += deal.flag * deal.quantity
            else:
                pos[deal.symbol] = deal.flag * deal.quantity
        finished_pos_symbols = []
        for symbol, quant in pos.items():
            if quant == 0:
                finished_pos_symbols.append(symbol)
        for symbol in finished_pos_symbols:
            pos.pop(symbol)
        return pos

    def equity_allocation(self, date):
        """
        Compute equity allocation status up to `date` in currency

        Return
        ------
        (``float``, ``float``)
            First tuple element is the long and second the short positions

        """
        pos = self.symbol_positioning(date)
        long = 0
        short = 0
        for symbol, pos_quant in pos.items():
            price = self.__approx_price(symbol, date)
            if pos_quant >= 0:
                long += price * pos_quant
            else:
                short += price * (-pos_quant)
        return long, short

    def net_equity(self, date):
        """
        Return net result if all operations are close. Negative means debt
        """
        long, short = self.equity_allocation(date)
        return long - short

    def overall_result(self, date):
        """
        Compute the result of the portfolio up to some `date`
        This include the profit/loss of all trades looking in
        the portfolio records and the net result of the open
        positions (long and short)

        Parameters
        ---
        `date` : ``pandas.Timestamp``
            Date to consider in result evaluation

        Return
        ---
        ``float``
            Net result from trades and open positions in currency

        """
        deals_stack = {}
        result = 0
        for deal in self.get_deals_before(date):
            if deal.symbol not in deals_stack.keys():
                # means there is no open position for this symbol
                deals_stack[deal.symbol] = [deal]
                continue
            if deals_stack[deal.symbol][0].flag * deal.flag > 0:
                # means the new deal raised the current position
                deals_stack[deal.symbol].append(deal)
            else:
                # means counter order from current position
                quant = deal.quantity
                while deals_stack[deal.symbol]:
                    if quant - deals_stack[deal.symbol][-1].quantity < 0:
                        break
                    pop_deal = deals_stack[deal.symbol].pop()
                    quant -= pop_deal.quantity
                    result += pop_deal.net_result(
                        deal.unit_price, deal.deal_date
                    )
                if deals_stack[deal.symbol]:
                    result += deals_stack[deal.symbol][-1].partial_close(
                        deal.unit_price, quant, deal.deal_date
                    )
                else:
                    # list of deals became empty for this symbol
                    deals_stack.pop(deal.symbol)
                    if quant > 0:
                        # reverse position
                        new_deal = StockDeal(
                            deal.symbol,
                            quant,
                            deal.unit_price,
                            deal.deal_date,
                            deal.flag,
                            deal.fixed_tax,
                            deal.relative_tax,
                            deal.daily_tax,
                        )
                        deals_stack[deal.symbol] = [new_deal]
        for symbol, deals_list in deals_stack.items():
            approx_price = self.__approx_price(symbol, date)
            for deal in deals_list:
                result += deal.net_result(approx_price, date)
        return result

    def symbols_result_time_series(
        self, time_step="day", stop_date=None, as_dataframe=False
    ):
        """
        Compute portfolio result evolution along time for each symbol

        Parameters
        ---
        `time_step` : ``int`` or "day"
            available time step to consider for each candle stick
        `stop_date` : ``pandas.Timestamp``
            datetime to stop the analysis. Default `self.portfolio_end`
        `as_dataframe` : ``bool``
            if True return a ``pandas.DataFrame`` columns label by symbols

        Return
        ---
        ``dict`` {`symbol` : `result_series`}
            `symbol` : valid code of shares in stock market
            `result_series` : result along time
        or
        ``pandas.DataFrame``
            columns label by symbols

        """
        if self.empty_history():
            raise IOError("Empty portfolio")
        if stop_date is None:
            stop_date = self.portfolio_end
        deal_events = self.get_deals_before(stop_date)
        if not deal_events:
            raise ValueError("There are no deals up to {}".format(stop_date))
        deals_stack = {}
        stack_status = {}
        sym_res = {}
        trades = {}
        invest_series = pd.Series([], dtype=float)
        for i, deal in enumerate(deal_events):
            if deal.symbol not in sym_res.keys():
                sym_res[deal.symbol] = pd.Series([], dtype=float)
                trades[deal.symbol] = 0.0
            if deal.symbol not in deals_stack.keys():
                # means there is no open position for this symbol
                price = deal.unit_price
                quant = deal.quantity
                deals_stack[deal.symbol] = [deal]
                stack_status[deal.symbol] = (quant, price)
            elif deals_stack[deal.symbol][0].flag * deal.flag > 0:
                # means the new deal raised the current position
                prev_quant = stack_status[deal.symbol][0]
                prev_price = stack_status[deal.symbol][1]
                quant = prev_quant + deal.quantity
                mean_price = (
                    prev_quant * prev_price + deal.unit_price * deal.quantity
                ) / quant
                deals_stack[deal.symbol].append(deal)
                stack_status[deal.symbol] = (quant, mean_price)
            else:
                # means counter order from current position
                quant = deal.quantity
                stack_flag = deals_stack[deal.symbol][0].flag
                old_quant = stack_status[deal.symbol][0]
                old_price = stack_status[deal.symbol][1]
                trade_quant = min(quant, old_quant)
                raw_trade_result = (
                    trade_quant * stack_flag * (deal.unit_price - old_price)
                )
                trade_taxes = 0
                while deals_stack[deal.symbol]:
                    if quant - deals_stack[deal.symbol][-1].quantity < 0:
                        stack_top = deals_stack[deal.symbol][-1]
                        trade_taxes += stack_top.total_taxes(
                            deal.deal_date, quant
                        )
                        stack_top.quantity -= quant
                        stack_status[deal.symbol] = (
                            old_quant - deal.quantity,
                            old_price,
                        )
                        quant = 0
                        break
                    pop_deal = deals_stack[deal.symbol].pop()
                    trade_taxes += pop_deal.total_taxes(deal.deal_date)
                    quant -= pop_deal.quantity
                if not deals_stack[deal.symbol]:
                    # list of deals became empty for this symbol
                    deals_stack.pop(deal.symbol)
                    stack_status.pop(deal.symbol)
                    if quant > 0:
                        # reverse position
                        new_deal = StockDeal(
                            deal.symbol,
                            quant,
                            deal.unit_price,
                            deal.deal_date,
                            deal.flag,
                            deal.fixed_tax,
                            deal.relative_tax,
                            deal.daily_tax,
                        )
                        deals_stack[deal.symbol] = [new_deal]
                        stack_status[deal.symbol] = (quant, deal.unit_price)
                trades[deal.symbol] += raw_trade_result - trade_taxes
            last_deal_date = deal.deal_date
            if i + 1 < len(deal_events):
                next_deal_date = deal_events[i + 1].deal_date
            else:
                next_deal_date = stop_date
            time_index = (
                self.refined_obj[deal.symbol]
                .time_bars(step=time_step)
                .loc[last_deal_date:next_deal_date]
                .index
            )
            open_position_res = {}
            open_position_invest = {}
            for symbol, (quant, mean_price) in stack_status.items():
                position_flag = deals_stack[symbol][0].flag
                price_series = (
                    self.refined_obj[symbol]
                    .time_bars(step=time_step)
                    .Close.loc[last_deal_date:next_deal_date]
                )
                if price_series.empty:
                    continue
                if position_flag > 0:
                    raw_invest_period = pd.Series(
                        quant * mean_price * np.ones(price_series.size),
                        price_series.index,
                    )
                else:
                    raw_invest_period = quant * price_series
                raw_result_series = (
                    quant * position_flag * (price_series - mean_price)
                )
                const_tax = (
                    len(deals_stack[symbol]) * self.fixed_tax
                    + self.relative_tax * mean_price * quant
                )
                roll_tax = np.zeros(price_series.size)
                if position_flag < 0:
                    for short_deal in deals_stack[symbol]:
                        roll_tax += short_deal.time_rolling_tax(
                            price_series.index
                        ).values
                accum_roll_tax_series = const_tax + pd.Series(
                    roll_tax, index=price_series.index
                )
                open_position_res[symbol] = (
                    raw_result_series - accum_roll_tax_series
                )
                open_position_invest[symbol] = (
                    raw_invest_period + accum_roll_tax_series
                )
            total_invest = pd.Series(np.zeros(time_index.size), time_index)
            for invest in open_position_invest.values():
                total_invest = total_invest.add(invest, fill_value=0)
            invest_series = invest_series.append(total_invest)
            for symbol in sym_res.keys():
                trade_ser = pd.Series(
                    trades[symbol] * np.ones(time_index.size), time_index
                )
                if symbol in open_position_res.keys():
                    sym_res[symbol] = sym_res[symbol].append(
                        trade_ser + open_position_res[symbol]
                    )
                else:
                    sym_res[symbol] = sym_res[symbol].append(trade_ser)
        if not as_dataframe:
            return sym_res
        df = pd.DataFrame(sym_res)
        df["NET_RESULT"] = df.sum(axis=1, skipna=True)
        df["INVESTMENT"] = invest_series
        return df.replace(np.nan, 0)
