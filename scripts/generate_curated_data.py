import os
import mvp
import argparse


def check_keys(p):
    expected_keys = ["MA", "DEV", "RSI", "AC_WINDOW", "AC_SHIFT_MAX"]
    for k in p.keys():
        if k not in expected_keys:
            raise argparse.ArgumentTypeError(
                "The key {} is not expected. Use one of following keys\n"
                "\t{}".format(k, expected_keys)
            )


def parameters(s):
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
            "\tKEY1:V11,V12,...,V1N:[KEY2:V21,V22,...,V2N]\n"
            "where [...] is optional."
        )
    check_keys(p)
    return p


def run(db_path, parameters, daily_option):
    """
    This function is a script that gives us a collection of objects containing the statistical features
    related to the raw data extracted from public sources. These statistical features are:
    - Moving Average
    - Standard Deviations
    - Relative Strength Index(RSI)
    - Autocorrelation
    - Fractionally Differentiated Series
    - ADF Test

    Parameters
    ----------
    `db_path` : ``str``
        full path to database file

    `parameters` : `` dict ``
        Dictionary with features as strings in keys and the
        evaluation feature paramter as values or list of values
        The (keys)strings corresponding to features must be:
        "MA" = Moving Average
        "DEV" = Standart Deviation
        "RSI" = Relative Strenght Index (RSI) indicator
        "AC_WINDOW": Observation window using the last point as reference
        "AC_SHIFT_MAX": The max value of lag values list (using step=1)

    `daily_option` : `` bool `` (optional)
        Automatically convert 1-minute raw data to daily data

    Return
    ------
    `dict_curated_data_objects` : ``dict``
        dictionary of CuratedData objects with SYMBOL as key values.
    """
    if not os.path.isfile(db_path):
        raise IOError("Database file {} not found".format(db_path))

    try:
        parameters["AC_SHIFT_MAX"] = list(
            range(1, parameters["AC_SHIFT_MAX"][0])
        )
    except:
        pass

    symbols = mvp.helper.get_db_symbols(db_path)[1:]
    list_raw_data_objects = []
    for symbol in symbols[:1]:
        temp = mvp.rawdata.RawData(
            symbol,
            db_path,
        )
        list_raw_data_objects.append(temp)
    dict_curated_data_objects = {}
    for raw_data in list_raw_data_objects:
        temp = mvp.curated.CuratedData(
            raw_data, parameters, daily=daily_option
        )
        dict_curated_data_objects[temp.symbol] = temp
        print(dict_curated_data_objects)
    return dict_curated_data_objects


if __name__ == "__main__":
    root_dir = os.path.join(os.path.expanduser("~"), "FintelligenceData")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--db-path",
        type=str,
        default=os.path.join(root_dir, "MetaTrader_M1.db"),
        help="Path to database file",
    )
    p.add_argument(
        "--parameters",
        type=parameters,
        default="MA:10:DEV:10:RSI:4,14,30:AC_WINDOW:100,200,300:AC_SHIFT_MAX:11",
        help="Parameters for the feature columns using the following convention:\n"
        "\tKEY1:V11,V12,...,V1N:[KEY2:V21,V22,...,V2N]\n"
        "where [...] is optional.",
    )
    p.add_argument(
        "--daily-option",
        action="store_true",
        help="Indicate the use of date in daily resolution",
    )
    args = p.parse_args()
    run(**vars(args))
