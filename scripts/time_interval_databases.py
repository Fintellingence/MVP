import os
import sqlite3
import argparse
from mvp import rawdata


def skip_db_msg(db_path):
    print("Skipping database {}. It already exists".format(db_path))


def get_db_version(db_path):
    return db_path.split("_")[-1].split(".")[0]


def script_run(raw_db_path, intervals, day, test):
    """
    Build databases for other intervals than 1-minute
    By default the new databases files are set in the same directory

    Paramters
    ---------
    `raw_db_path` : `str`
        full path with 1-minute database
    `intervals` : `list`(`int`)
        new time intervals in minutes. Must be [5, 10, 15, 30, 60]
    `day` : `boolean`
        whether to also build the daily database or not
    `test` : `boolean`
        If True, run script for 3 symbols and delete files afterwards

    """
    if not os.path.isfile(raw_db_path):
        raise IOError("{} file does not exist".format(raw_db_path))
    version = get_db_version(raw_db_path)
    base_dir = os.path.dirname(raw_db_path)
    file_interval_dict = dict(
        map(
            lambda x: ("minute{}_database_{}.db".format(x, version), x),
            intervals,
        )
    )
    conn_dict = {}
    for file_name in file_interval_dict.keys():
        full_file_path = os.path.join(base_dir, file_name)
        if not os.path.isfile(full_file_path):
            conn_dict[file_name] = sqlite3.connect(full_file_path)
        else:
            skip_db_msg(full_file_path)
    if day:
        file_name = "daily_database_{}.db".format(version)
        file_interval_dict[file_name] = "day"
        full_file_path = os.path.join(base_dir, file_name)
        if not os.path.isfile(full_file_path):
            conn_dict[file_name] = sqlite3.connect(full_file_path)
        else:
            skip_db_msg(full_file_path)
    if len(conn_dict.keys()) == 0:
        return
    db_symbols = rawdata.get_db_symbols(raw_db_path)
    for i, symbol in enumerate(db_symbols):
        print("[{:2d}/{}] {}".format(i + 1, len(db_symbols), symbol))
        symbol_data = rawdata.RawData(symbol, raw_db_path)
        for file_key in conn_dict.keys():
            df = symbol_data.change_sample_interval(
                step=file_interval_dict[file_key]
            )
            df.to_sql(symbol, con=conn_dict[file_key])
        if test and i > 2:
            break
    for file_key in conn_dict.keys():
        conn_dict[file_key].close()
    if test:
        for file_name in conn_dict.keys():
            os.remove(os.path.join(base_dir, file_name))


if __name__ == "__main__":
    default_dir = os.path.join(os.path.expanduser("~"), "FintelligenceData")
    p = argparse.ArgumentParser(
        usage="python %(prog)s -[arguments] ",
        description="Build databases with requested intervals",
    )
    p.add_argument(
        "--raw-db-path",
        dest="raw_db_path",
        type=str,
        default=os.path.join(default_dir, "minute1_database_v1.db"),
        help="path to (raw) database file with 1-minute data",
    )
    p.add_argument(
        "-intervals",
        nargs="+",
        type=int,
        default=[],
        help="list of new intervals to reassemble the database",
    )
    p.add_argument(
        "-day",
        action="store_true",
        help="Boolean to assemble daily bar database",
    )
    p.add_argument(
        "-test",
        action="store_true",
        help="Boolean to run a test using only 3 symbols",
    )
    args = p.parse_args()
    script_run(**vars(args))
