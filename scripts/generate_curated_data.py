import os
import mvp
import numpy as np
import argparse
import ast


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

    """
    This first section is for handling error regarding to failed parameters inputs via terminal.
    """
    if not os.path.isfile(db_path):
        raise IOError("Database file {} not found".format(db_path))

    parameters_dict = parameters
    if type(parameters) == str:
        parameters_dict = ast.literal_eval(parameters)
        if len(parameters_dict) == 0:
            raise IOError("Parameters are needed in order to continue")
        try:
            parameters_dict["AC_SHIFT_MAX"] = list(
                range(1, parameters_dict["AC_SHIFT_MAX"][0])
            )
        except:
            pass

    if type(daily_option) == str:
        if daily_option == "True":
            daily_option = ast.literal_eval(daily_option)
        elif daily_option == "False":
            daily_option = ast.literal_eval(daily_option)
        else:
            raise IOError("Daily option needs to be Boolean")

    """
    This section is focused in generating curated data from a .db file providing \
        us curated data including new features like Moving Averages, Standard Deviations and RSI indicator. 
    """
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
            raw_data, parameters_dict, daily=daily_option
        )
        dict_curated_data_objects[temp.symbol] = temp

    return dict_curated_data_objects


if __name__ == "__main__":
    root_dir = os.path.join(os.path.expanduser("~"), "FintelligenceData")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--db-path",
        dest="db_path",
        type=str,
        default=os.path.join(root_dir, "MetaTrader_M1.db"),
        help="path to database file",
    )
    p.add_argument(
        "--features-parameters",
        dest="parameters",
        type=str,
        default={
            "MA": [10],
            "DEV": [10],
            "RSI": [5, 14, 30],
            "AC_WINDOW": [100, 200, 300],
            "AC_SHIFT_MAX": list(range(1, 11)),
        },
        help="parameters for the feature columns using the following convention: \
        \"{'MA':[PERIODS],'DEV':[PERIODS],'RSI':[PERIODS],'AC_WINDOW':[WINDOWS_TO_LOOK],'AC_SHIFT_MAX':[SHIFT_MAX]}\" ",
    )
    p.add_argument(
        "--daily-option",
        dest="daily_option",
        type=str,
        default=False,
        help="Choose if you want to see the data in daily resolution or not: Boolean Variable(True or False))",
    )
    args = p.parse_args()
    run(**vars(args))