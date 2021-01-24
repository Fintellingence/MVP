import os
import mvp
import numpy as np
import argparse
import ast


def run(db_path, sym_path, parameters):
    """
    This first section is for handling error regarding to failed parameters inputs via terminal.
    """
    if not os.path.isfile(db_path):
        raise IOError("Database file {} not found".format(db_path))
    if not os.path.isfile(sym_path):
        raise IOError("Symbols txt file {} not found".format(sym_path))

    parameters_dict = parameters
    if type(parameters) == str:
        parameters_dict = ast.literal_eval(parameters)
        if len(parameters_dict) == 0:
            raise IOError("Parameters needed to continue")

    """
    This section is focused in generating curated data from a .db file and a "symbols" text file providing \
        us curated data including new features like Moving Averages, Standard Deviations and RSI indicator. 
    """
    symbols = mvp.helper.get_symbols(sym_path)[1:]

    list_raw_data_objects = []
    for symbol in symbols[:1]:
        temp = mvp.rawdata.RawData(symbol, db_path)
        list_raw_data_objects.append(temp)

    list_curated_data_objects = []
    for raw_data in list_raw_data_objects:
        temp = mvp.curated.CuratedData(raw_data, parameters_dict)
        list_curated_data_objects.append(temp)

    # TODO: generate a new .db or update the existing .db. For now, the code is simply printing the result and returning None.
    print(list_curated_data_objects[0].df_curated.tail(20))
    print(list_curated_data_objects[0].autocorr_values)
    return None


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
        "--sym-path",
        dest="sym_path",
        type=str,
        default=os.path.join(root_dir, "stocks.txt"),
        help="path to symbols txt file",
    )
    p.add_argument(
        "--features-parameters",
        dest="parameters",
        type=str,
        default={
            "MA": [10],
            "DEV": [10],
            "RSI": [5, 15],
            "AC_WINDOW": [100, 200, 300, 400, 500],
            "AC_SHIFT": [5, 10, 15, 20],
        },
        help="parameters for the feature columns using the following convention: \
        \"{'MA':[PERIODS],'DEV':[PERIODS],'RSI':[PERIODS],'AC_WINDOW':[WINDOWS_TO_LOOK],'AC_SHIFT':[SHIFTS]}\" ",
    )
    args = p.parse_args()
    run(**vars(args))