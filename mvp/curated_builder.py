import os
import mvp
import argparse

ROOT_DIR = os.path.join(os.path.expanduser("~"), "FintelligenceData")
DB_PATH_FINTELLIGENCE = os.path.join(ROOT_DIR, "minute1_database_v1.db")


class CuratedSet:
    """
    This class contains a collection of objects
    containing the statistical features related to the raw data
    extracted from public sources. These statistical features are:
    - Moving Average
    - Standard Deviations
    - Relative Strength Index(RSI)
    - Autocorrelation
    - Fractionally Differentiated Series
    - ADF Test

    These features are methods in `curated.py` class. Some methods are
    initialized with the class, and others are called when needed.

    Parameters
    ----------
    `db_path` : ``str``
        full path to database file

    `parameters` : `` str ``
        String containing keys and values for evaluation of
        the statistical features.
        The (keys)strings corresponding to features must be:
        "MA" = Moving Average
        "DEV" = Standart Deviation
        "RSI" = Relative Strenght Index (RSI) indicator

        Feature parameters are given using the following
        convention:

        "KEY1:V11,V12,...:KEY2:V21,V22,...V2N"

    `daily_option` : `` bool `` (optional)
        Automatically convert 1-minute raw data to daily data

    Return
    ------
    `dict_curated_data_objects` : ``dict``
        dictionary of CuratedData objects with SYMBOL as key values.

    """

    def __init__(
        self,
        db_path=DB_PATH_FINTELLIGENCE,
        parameters="MA:10:DEV:10:RSI:4,14,30",
        daily_option=True,
    ):

        if not os.path.isfile(db_path):
            raise IOError("Database file {} not found".format(db_path))
        self.db_path = db_path
        self.parameters = self.parameters_parse(parameters)
        self.daily_option = daily_option
        self.symbols = mvp.rawdata.get_db_symbols(db_path)
        self.dict_raw_data_objects = self.build_raw_data_objects()
        self.dict_curated_data_objects = self.build_curated_data_objects()

    def check_keys(self, p):
        expected_keys = ["MA", "DEV", "RSI"]
        for k in p.keys():
            if k not in expected_keys:
                raise argparse.ArgumentTypeError(
                    "The key {} is not expected. Use one of following keys\n"
                    "\t{}".format(k, expected_keys)
                )

    def parameters_parse(self, s):
        p = {}
        s = s.strip()
        if s[-1] == ":":
            s = s[:-1]
        try:
            key_values_str = s.split(":")
            if len(key_values_str) % 2 != 0:
                raise argparse.ArgumentTypeError(
                    "All keys must have a associated value"
                )
            keys = key_values_str[::2]
            values = key_values_str[1::2]
            for k, v in zip(keys, values):
                p[k] = list(map(int, v.split(",")))
        except:
            raise argparse.ArgumentTypeError(
                "Parameters must be inserted as\n"
                "\tKEY1:V11,V12,...,V1N:KEY2:V21,V22,...\n"
            )
        self.check_keys(p)
        return p

    def build_raw_data_objects(self):
        dict_raw_data_objects = {}
        for symbol in self.symbols:
            temp = mvp.rawdata.RawData(
                symbol,
                self.db_path,
            )
            dict_raw_data_objects[symbol] = temp
        return dict_raw_data_objects

    def build_curated_data_objects(self):
        dict_curated_data_objects = {}
        for raw_data in self.dict_raw_data_objects:
            temp = mvp.curated.CuratedData(
                self.dict_raw_data_objects[raw_data],
                self.parameters,
                daily=self.daily_option,
            )
            dict_curated_data_objects[temp.symbol] = temp
        return dict_curated_data_objects