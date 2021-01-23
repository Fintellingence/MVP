import os
import mvp
import numpy as np
import argparse


def run(db_path, sym_path, parameters):
    if not os.path.isfile(db_path):
        raise IOError("Database file {} not found".format(db_path))
    if not os.path.isfile(sym_path):
        raise IOError("symbols txt file {} not found".format(sym_path))
    symbols = mvp.helper.get_symbols(sym_path)[1:]

    list_raw_data_objects = []
    for symbol in symbols[:1]:
        temp = mvp.rawdata.RawData(symbol, db_path)
        list_raw_data_objects.append(temp)

    list_curated_data_objects = []
    for raw_data in list_raw_data_objects:
        temp = mvp.curated.CuratedData(raw_data, parameters)
        list_curated_data_objects.append(temp)

    # TODO: generate a new .db or update the existing .db. For now, the code is simply printing the result.
    return print(list_curated_data_objects[0].df_curated.tail(20))


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
        "--features parameters",
        dest="parameters",
        type=dict,
        default={
            "MA": [10],
            "DEV": [10],
            "RSI": [5, 15],
        },
        help="parameters for the feature columns using the following convention: \
        {'MA':[PERIODS],'DEV':[PERIODS],'RSI':[PERIODS]'}",
    )
    args = p.parse_args()
    run(**vars(args))