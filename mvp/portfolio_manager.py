import os

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
        common_features="MA_DAY:10,20:DEV_DAY:10,20:MA_60:6:DEV_60:6",
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

            with the following data types of `Vij` in between parentheses

            Note the underscore after KEYj which can be one of the following
            1, 5, 10, 15, 30, 60 and "DAY" indicating the time step to be
            used in resampling the data to evaluare the statistical features

            example
            -------
            "MA_60:100,1000:DEV_DAY:10,20:FRAC_DIFF_DAY:1,0.5"

            compute for all symbols introduced in this object the moving
            average for 60-minute bars with windows of 100, 1000, the
            moving standard deviation for daily-minute bars using 10 and
            20 days. Finally the fractional differentiation for 1 (daily
            returns) and 0.5.

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
        key_interval_codes = input_string.split(":")[::2]
        string_intervals = [key.split("_")[1] for key in key_interval_codes]
        intervals = []
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
        key_value_list_split = self.raw_input_string.split(":")
        keys = key_value_list_split[::2]
        str_vals = key_value_list_split[1::2]
        input_dict = {}
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
            "MA_60:100,1000:DEV_DAY:10,20:FRAC_DIFF_DAY:1,0.5"

        """
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
            return

    def reset_common_features(self, new_input_string):
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
            "MA_60:100,1000:DEV_DAY:10,20:FRAC_DIFF_DAY:1,0.5"

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
            return

    def new_symbol(self, symbol, start=None, stop=None):
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

    def remove_symbol(self, symbol):
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
