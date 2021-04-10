import os
import numpy as np
import pandas as pd
import bisect

from functools import total_ordering
from mvp.refined_data import RefinedData

ROOT_DIR = os.path.join(os.path.expanduser("~"), "FintelligenceData")
DB_PATH_FINTELLIGENCE = os.path.join(ROOT_DIR, "minute1_database_v1.db")


class RefinedSet:
    """
    Class to collect a set of stock shares and obtain common features
    For more info about the features see `RefinedData` class. Despite
    the possibility to compute new features acessing refined objects
    by the `refined_obj` attribute, it is advisable to restricted the
    analysis to the common features requested in the constructor, since
    they are set in cache memory.

    Main attributes
    ---------------
    `refined_obj` : ``dict {str : RefinedData}``
        set of refined data objects acessed using the symbol as key
    `symbol_period` = ``dict {str : (pandas.Timestamp, pandas.Timestamp)}``
        (start, stop) with period the refined object was analyzed

    """

    def __init__(
        self,
        db_path=DB_PATH_FINTELLIGENCE,
        common_features="MA_DAY:10,20:DEV_DAY:10,20",
        preload={"time": [5, 10, 15, 30, 60, "day"]},
    ):
        """
        Parameters
        ----------
        `db_path` : ``str``
            full path to 1-minute database file

        `common_features` : ``str``
            All information of features to compute for all companies loaded
            Must be formatted as follows:

            "KEY1_T1:V11,V12,...:KEY2_T2:V21,V22,...:...:KEYM_TM:VM1,..."

            where KEY is an `RefinedData` method abbreviation. Must be one of

            "MA" = Moving Average (``int``)
            "DEV" = Standart Deviation (``int``)
            "RSI" = Relative Strenght Index (RSI) indicator (``int``)
            "FRACDIFF": Fractional differentiation (``float``)

            with the following data types of `Vij` in parentheses

            Note the underscore after KEYj which can be one of the following
            1, 5, 10, 15, 30, 60 and DAY indicating the time step to be used
            in resampling the data to evaluare the statistical features

            example
            -------
            "MA_60:100,1000:DEV_DAY:10,20:FRAC_DIFF_DAY:1,0.5"

            compute for all symbols introduced in this object the moving
            average for 60-minute bars with windows of 100, 1000, the
            moving standard deviation for daily bars using 10 and 20 days
            Finally the fractional differentiation for 1 (daily returns)
            and 0.5.

            In this string no successive : is allowed as well as : at the
            end or beginning. Colons must aways be surrounded by keys and
            values

        `preload` : ``dict``
            inform dataframes time intervals to hold in cache

        """
        if not os.path.isfile(db_path):
            raise IOError("Database file {} not found".format(db_path))
        self.db_path = db_path
        self.__refined_attr = {
            "MA": "get_simple_MA",
            "DEV": "get_deviation",
            "RSI": "get_RSI",
            "FRACDIFF": "frac_diff",
        }
        self.available_time_intervals = [1, 5, 10, 15, 30, 60, "day"]
        self.preload_data = preload
        self.raw_input_string = self.__assert_input_string(common_features)
        self.time_intervals = self.__extract_intervals(self.raw_input_string)
        self.input_dict = self.__convert_input_dict()
        self.refined_obj = {}
        self.symbol_period = {}

    def __assert_input_string(self, input_string):
        """
        validade the input string with common features

        Return
        ------
        ``str``
            `input_string` removing repetitions and sorting the parameters
        """
        if not input_string:
            return input_string
        key_value_list_split = input_string.split(":")
        if len(key_value_list_split) % 2 != 0:
            raise ValueError(
                "Wrong pairings divided by colons in {}".format(input_string)
            )
        keys = key_value_list_split[::2]
        str_vals = key_value_list_split[1::2]
        for key, str_val in zip(keys, str_vals):
            if not str_val or not key:
                raise ValueError(
                    "empty fields separated by : in {}".format(input_string)
                )
            if len(key.split("_")) != 2:
                raise ValueError(
                    "Each key-code must have only one '_' separating "
                    "feature abbreviation and interval. Check {}".format(
                        input_string
                    )
                )
            attr_abbr_key = key.split("_")[0]
            if attr_abbr_key not in self.__refined_attr.keys():
                raise ValueError(
                    "Found invalid abbreviation {} in {}".format(
                        attr_abbr_key, input_string
                    )
                )
        intervals = self.__extract_intervals(input_string)
        if not set(intervals).issubset(set(self.available_time_intervals)):
            raise ValueError(
                "There are invalid time intervals in {}".format(input_string)
            )
        return self.__remove_repetitions(input_string)

    def __extract_intervals(self, input_string):
        """
        Return list with appropriate interval datatypes from input string
        """
        intervals = []
        if not input_string:
            return intervals
        key_interval_codes = input_string.split(":")[::2]
        string_intervals = [key.split("_")[1] for key in key_interval_codes]
        for string_interval in string_intervals:
            try:
                time_interval = int(string_interval)
            except ValueError:
                time_interval = string_interval.lower()
            if time_interval not in intervals:
                intervals.append(time_interval)
        return intervals

    def __remove_repetitions(self, input_string):
        """
        Remove any duplicated feature of `input_string`
        and sort parameters in ascending order
        """
        key_value_list_split = input_string.split(":")
        keys = key_value_list_split[::2]
        str_vals = key_value_list_split[1::2]
        no_rep_map = {}
        for key, str_val in zip(keys, str_vals):
            if key in no_rep_map.keys() and str_val:
                raw_str_val = no_rep_map[key] + "," + str_val
            else:
                raw_str_val = str_val
            try:
                num_vals = list(map(int, set(raw_str_val.split(","))))
            except Exception:
                num_vals = list(map(float, set(raw_str_val.split(","))))
            num_vals.sort()
            str_val_unique = ",".join(map(str, num_vals))
            no_rep_map[key] = str_val_unique
        no_rep_list = []
        for no_rep_key, no_rep_val in no_rep_map.items():
            no_rep_list.append(no_rep_key)
            no_rep_list.append(no_rep_val)
        no_rep_input_string = ":".join(no_rep_list)
        return no_rep_input_string

    def __convert_input_dict(self):
        """
        Process the input (raw)string with features to be computed

        Return
        ------
        ``dict`` {"KEY_T" : [ value1, value2, ... , valueN }
            "KEY" is a method abbreviation and "T" a period. All
            values are mapped to number datatypes instead of strings

        """
        input_dict = {}
        if not self.raw_input_string:
            return input_dict
        key_value_list_split = self.raw_input_string.split(":")
        keys = key_value_list_split[::2]
        str_vals = key_value_list_split[1::2]
        for key, str_val in zip(keys, str_vals):
            try:
                values_list = list(map(int, str_val.split(",")))
            except ValueError:
                values_list = list(map(float, str_val.split(",")))
            input_dict[key] = values_list
        return input_dict

    def __clean_features_cache(self):
        for ref_obj in self.refined_obj.values():
            ref_obj.cache_clean()

    def is_empty(self):
        """
        Return `True` if there are no symbols refined in this object
        """
        return not self.refined_obj

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

    def display_info(self):
        """
        Print on screen current status of this symbol set object
        """
        print("\nActual status of refined set\n")
        for symbol in self.refined_obj.keys():
            start = self.symbol_period[symbol][0]
            stop = self.symbol_period[symbol][1]
            print("{} from {} to {}".format(symbol, start, stop))
        print("\nraw input : {}".format(self.raw_input_string))
        print("\nFeature key : list of parameters used")
        for input_key in self.input_dict.keys():
            print("\n{:20s}".format(input_key), end=" ")
            for value in self.input_dict[input_key]:
                print(value, end=" ")
        print()

    def add_common_features(self, new_input_string):
        """
        Append `new_input_string` in features common to all symbols

        Parameters
        ----------
        `new_input_string` : ``str``
            All information of features to compute for all companies loaded

            "KEY1_T1:V11,V12,...:KEY2_T2:V21,V22,...:...:KEYM_TM:VM1,..."

            where KEY is an `RefinedData` method abbreviation. Must be one of

            "MA" = Moving Average (``int``)
            "DEV" = Standart Deviation (``int``)
            "RSI" = Relative Strenght Index (RSI) indicator (``int``)
            "FRACDIFF": Fractional differentiation (``float``)

            with the following data types of `Vij` in between parentheses

            Note the underscore after KEYj which can be one of the following
            1, 5, 10, 15, 30, 60 and "DAY" indicating the time step to be
            used in resampling the data to evaluare the statistical features

            example
            -------
            "MA_60:100,1000:DEV_DAY:10,20:FRACDIFF_DAY:1,0.5"

        """
        if not new_input_string:
            return
        try:
            self.raw_input_string = self.__assert_input_string(
                self.raw_input_string + ":" + new_input_string
            )
            self.input_dict = self.__convert_input_dict()
            self.refresh_all_features()
        except ValueError as err:
            print(
                err,
                "An error occurred while appending the new string {}".format(
                    new_input_string
                ),
                sep="\n\n",
            )

    def reset_common_features(self, new_input_string):
        """
        Append `new_input_string` in features common to all symbols

        Parameters
        ----------
        `new_input_string` : ``str``
            All information of features to compute for all companies loaded
            if empty string given it will just clean the features cache

            "KEY1_T1:V11,V12,...:KEY2_T2:V21,V22,...:...:KEYM_TM:VM1,..."

            where KEY is an `RefinedData` method abbreviation. Must be one of

            "MA" = Moving Average (``int``)
            "DEV" = Standart Deviation (``int``)
            "RSI" = Relative Strenght Index (RSI) indicator (``int``)
            "FRACDIFF": Fractional differentiation (``float``)

            with the following data types of `Vij` in between parentheses

            Note the underscore after KEYj which can be one of the following
            1, 5, 10, 15, 30, 60 and "DAY" indicating the time step to be
            used in resampling the data to evaluare the statistical features

            example
            -------
            "MA_60:100,1000:DEV_DAY:10,20:FRACDIFF_DAY:1,0.5"

        """
        try:
            self.__clean_features_cache()
            self.raw_input_string = self.__assert_input_string(
                new_input_string
            )
            self.input_dict = self.__convert_input_dict()
            self.refresh_all_features()
        except ValueError as err:
            print(
                err,
                "An error occurred with the new string {}".format(
                    new_input_string
                ),
                sep="\n\n",
            )

    def new_refined_symbol(self, symbol, start=None, stop=None):
        """
        Introduce new symbol in the set for a given period. The period
        refers to the common features that are computed and further
        used as reference to portfolio management.

        Parameters
        ----------
        `symbol` : ``str``
            valid symbol contained in the `self.db_path` database
        `start` : ``pandas.Timestamp``
            date-time of inclusion in the set
        `stop` : ``pandas.Timestamp``
            date-time of exclusion in the set

        """
        if symbol in self.refined_obj.keys():
            return
        self.refined_obj[symbol] = RefinedData(
            symbol, self.db_path, preload=self.preload_data
        )
        valid_start, valid_stop = self.refined_obj[symbol].assert_window(
            start, stop
        )
        self.symbol_period[symbol] = (valid_start, valid_stop)
        for input_key in self.input_dict.keys():
            attr_abbr_key = input_key.split("_")[0]
            str_step = input_key.split("_")[1]
            try:
                time_step = int(str_step)
            except ValueError:
                time_step = str_step.lower()
            attr_name = self.__refined_attr[attr_abbr_key]
            for parameter in self.input_dict[input_key]:
                self.refined_obj[symbol].__getattribute__(attr_name)(
                    parameter, valid_start, valid_stop, time_step, True
                )

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
        Compute again all common features currently in `self.input_dict`
        """
        for symbol, ref_obj in self.refined_obj.items():
            start = self.symbol_period[symbol][0]
            stop = self.symbol_period[symbol][1]
            for input_key in self.input_dict.keys():
                attr_abbr_key = input_key.split("_")[0]
                str_step = input_key.split("_")[1]
                try:
                    time_step = int(str_step)
                except ValueError:
                    time_step = str_step.lower()
                attr_name = self.__refined_attr[attr_abbr_key]
                for parameter in self.input_dict[input_key]:
                    ref_obj.__getattribute__(attr_name)(
                        parameter, start, stop, time_step, True
                    )

    def correlation_matrix(self, time_step=1):
        """
        Correlation among all symbols in the set

        Parameters
        ---
        `time_step` : ``int`` or ``str``("day")
            time step of the dataframe sample

        Return
        ---
        ``pandas.DataFrame``

        """
        if time_step not in self.available_time_intervals:
            raise ValueError(
                "Time step {} not in available values {}".format(
                    time_step, self.available_time_intervals
                )
            )
        nsymbols = len(self.refined_obj.keys())
        mat = np.empty([nsymbols, nsymbols])
        items_pkg = self.refined_obj.items()
        for i, (symbol1, ref_obj1) in enumerate(items_pkg):
            for j, (symbol2, ref_obj2) in enumerate(items_pkg):
                if j < i:
                    continue
                start = max(
                    self.symbol_period[symbol1][0],
                    self.symbol_period[symbol2][0],
                )
                stop = min(
                    self.symbol_period[symbol1][1],
                    self.symbol_period[symbol2][1],
                )
                if stop <= start:
                    corr = np.nan
                else:
                    cls_prices1 = ref_obj1.change_sample_interval(
                        step=time_step
                    ).Close.loc[start:stop]
                    cls_prices2 = ref_obj2.change_sample_interval(
                        step=time_step
                    ).Close.loc[start:stop]
                    corr = cls_prices1.corr(cls_prices2)
                mat[i, j] = corr
                mat[j, i] = corr
        corr_df = pd.DataFrame(
            mat,
            columns=list(self.refined_obj.keys()),
            index=list(self.refined_obj.keys()),
            dtype=np.float64,
        )
        corr_df.name = "Correlation Matrix"
        corr_df.index.name = "symbols"
        return corr_df


@total_ordering
class StockDeal:
    """
    Object to represent stock market shares negociations with ordering
    methods based on date and time. Date variables assume ``pandas.Timestamp``

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
        Construct a deal using all required information

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
            contant value in currency to execute an order in the stock market
        `relative_tax` : ``float``
            fraction of the total order value
        `daily_tax` : ``float`` (default 0)
            fraction of share price charged to hold position per day
            Usually only applicable to maintain short position

        """
        self.symbol = symbol
        self.quantity = int(quantity)
        self.unit_price = unit_price
        self.deal_date = date
        self.daily_tax = daily_tax
        self.flag = flag
        self.fixed_tax = fixed_tax
        self.relative_tax = relative_tax
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
        if abs(self.flag) != 1:
            raise ValueError(
                "Flag in a deal must be 1>buy or -1>sell. {} given".format(
                    self.flag
                )
            )

    def __valid_input_date(self, date):
        return date > self.deal_date

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

    def total_rolling_tax(self, date, quantity):
        """ Compute the rolling cost of the operation up to `date` """
        if not self.__valid_input_date(date):
            return 0
        days_elapsed = 1 + (date - self.deal_date).days
        total_cost = quantity * self.unit_price
        return total_cost * days_elapsed * self.daily_tax

    def total_taxes(self, date, quantity):
        """ Return total amount spent with taxes up to `date` """
        if not self.__valid_input_date(date):
            return 0
        if quantity < self.quantity:
            inc_fixed = 0
        else:
            inc_fixed = self.fixed_tax
        return (
            inc_fixed
            + self.relative_tax * quantity * self.unit_price
            + self.total_rolling_tax(date, quantity)
        )

    def raw_result(self, date, unit_price, quantity):
        """ Return the raw result discounting the taxes up to `date` """
        if not self.__valid_input_date(date):
            return 0
        return quantity * (unit_price - self.unit_price) * self.flag

    def net_result(self, date, unit_price):
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
            return 0
        taxes = self.total_taxes(date, self.quantity)
        return self.raw_result(date, unit_price, self.quantity) - taxes

    def partial_close(self, date, unit_price, cls_quant):
        """
        Return partial result due to reduction in position
        The object internal quantity is changed reducing it by `cls_quant`

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
        cls_quant = int(cls_quant)
        if not self.__valid_input_date(date):
            return 0
        if cls_quant >= self.quantity:
            raise ValueError(
                "Unable to partial close. "
                "{} required but only {} in stock".format(
                    cls_quant, self.quantity
                )
            )
        taxes = self.total_taxes(date, cls_quant)
        op_result = self.raw_result(date, unit_price, cls_quant) - taxes
        self.quantity -= cls_quant
        return op_result


class PortfolioRecord:
    """
    Class to record a set of deals. Basically it holds as attribute a
    list of `StockDeal` objects ordered by date they ocurred. As date
    object all methods use to the `pandas.Timestamp`

    """

    def __init__(self):
        """ Initialize empty list of deals """
        self.__deals = []

    def empty_history(self):
        """ Return True if there are no deals registered """
        return not self.__deals

    def get_all_deals(self):
        """ Return the list of all deals recorded """
        return self.__deals.copy()

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

    def get_deals_before(self, date):
        """ Return list of deals objects occurred before `date` """
        i = 0
        for deal in self.__deals:
            if deal.deal_date >= date:
                break
            i = i + 1
        return self.__deals[:i].copy()

    def get_deals_after(self, date):
        """ Return list of deals objects occurred afeter `date` """
        i = 0
        for deal in self.__deals:
            if deal.deal_date > date:
                break
            i = i + 1
        return self.__deals[i:].copy()

    def delete_deals_symbol(self, symbol):
        """ Remove from records all deals related to the given `symbol` """
        for deal in self.__deals:
            if deal.symbol == symbol:
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
    def __init__(self, fixed_tax=0, relative_tax=0):
        self.fixed_tax = fixed_tax
        self.relative_tax = relative_tax
        RefinedSet.__init__(self, common_features="")
        PortfolioRecord.__init__(self)

    def __nearest_index(self, symbol, date):
        """ Return a valid integer index of the prices dataframe """
        return self.refined_obj[symbol].df.index.get_loc(
            date, method="nearest"
        )

    def __nearest_date(self, symbol, date):
        return self.refined_obj[symbol].df.index[
            self.__nearest_index(symbol, date)
        ]

    def __approx_price(self, symbol, date):
        return self.refined_obj[symbol].df.Close[
            self.__nearest_index(symbol, date)
        ]

    def set_position_buy(self, symbol, date, quantity):
        """
        Insert a buy(long) position in the portfolio

        Parameters
        ---
        `symbol` : ``str``
            share symbol in the stock market
        `date` : ``pandas.Timestamp``
            date the order was executed
        `quantity` : ``int`` (positive)
            How many shares to buy

        """
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

    def set_position_sell(self, symbol, date, quantity, daily_tax=0.001):
        """
        Register a sell order(short) in the portfolio

        Parameters
        ---
        `symbol` : ``str``
            share symbol in the stock market
        `date` : ``pandas.Timestamp``
            date the order was executed
        `quantity` : ``int`` (positive)
            How many shares to buy
        `dialy_tax` : ``float``
            the rent fraction required to maintain the order

        """
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
            if abs(quant) < 1:
                finished_pos_symbols.append(symbol)
        for symbol in finished_pos_symbols:
            pos.pop(symbol)
        return pos

    def equity_allocation(self, date):
        """
        Compute equity allocation status up to `date` in currency

        Return
        ---
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

    def overall_result(self, date, include_taxes=True):
        """
        Compute the result of the portfolio up to some `date`
        This include the profit/loss of all trades looking in
        the Portfolio records and the net equity of the open
        positions (long and short)

        Parameters
        ---
        `date` : ``pandas.Timestamp``
            Date to consider in result evaluation
        `include_taxes` : ``bool``
            If true(default) apply all taxes

        Return
        ---
        ``float``
            Net result from trades and open positions

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
                        deal.deal_date, deal.unit_price
                    )
                if deals_stack[deal.symbol]:
                    result += deals_stack[deal.symbol][-1].partial_close(
                        deal.deal_date, deal.unit_price, quant
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
                result += deal.net_result(date, approx_price)
        return result
