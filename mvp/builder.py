import os
import sqlite3
import logging
import logging.handlers

import pandas as pd
import datetime as dt
import pandas_datareader as pdr

from pathlib import Path

__all__ = ["Yahoo", "MetaTrader"]

# global variables to automate function calls
STD_DB_PATH = str(Path.home()) + "/FintelligenceData/BRShares_Yahoo_D1.db"
BIG_COMPANIES = ["PETR4","PETR3","VALE3","ITUB4",
                 "BBAS3","BBDC4","ITSA4","B3SA3"]

class Yahoo:
    """
    Define a database in Sqlite using Yahoo web API and pandas_datareader
    to collect shares OHLC prices in 1-Day time-frame.

    Parameters
    ----------
    db_path : ``str``
        The path to the dump file for a Sqlite database.

    """

    def __init__(self, db_path = STD_DB_PATH):
        os.makedirs("logs/", exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            "logs/yahoo.log", maxBytes=200 * 1024 * 1024, backupCount=1
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        self._logger = logging.getLogger("Logger")
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(handler)

        dir_name = os.path.dirname(db_path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._cursor = self._conn.cursor()
        self._std_init_day1 = dt.date(2010, 1, 2)
        self._db_symbols = self.get_db_symbols()

    def get_db_symbols(self):
        """
        Get all symbols for the connected Sqlite database.

        """
        table_names = self._cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        db_symbols = [name[0] for name in table_names]
        return db_symbols

    def get_date(self, symbol):
        """
        Get date associated with `symbol` plus one day.  If  `symbol`
        is not in database, return `self._std_init_day1` which is the
        default initial date for all companies.

        """
        if (symbol in self._db_symbols):
            full_date = self._cursor.execute(
                "SELECT Date FROM {} ORDER BY Date DESC LIMIT 1".format(symbol)
            ).fetchall()[0][0]
            date_str = full_date.split(" ")[0] # Take only YYYY-mm-dd part
            last_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
            init_day1 = last_date + dt.timedelta(days=1)
            return init_day1
        return self._std_init_day1

    def update_symbol(self, symbol, final_day1 = None):
        """
        Update data for the `symbol`. If the `symbol` is not in the
        database, introduce it starting from `self._std_init_day1`.
        `final_day1`(optional) must be datetime.date object. If not
        provided use today's date.

        """
        init_day1 = self.get_date(symbol)
        if (final_day1 == None):
            final_day1 = dt.date.today() - dt.timedelta(days=1)
        elif (type(final_day1) != dt.date):
            final_day1 = dt.date.today() - dt.timedelta(days=1)
            self._logger.warn("final date type of share {} is invalid. "
                    "Using today's date {}".format(symbols,final_day1))
        if (init_day1 + dt.timedelta(days=3) > final_day1):
            return 0    # flag for unecessary update
        flag = 1        # flag for update
        try:
            df = pdr.DataReader(symbol + ".SA", "yahoo", init_day1, final_day1)
            df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)
            df.to_sql(symbol, con=self._conn, if_exists="append")
            self._logger.info("{} successfully updated.".format(symbol))
            if (symbol not in self._db_symbols):
                self._db_symbols.append(symbol)
                flag = 2    # flag for new symbol
        except Exception as e:
            self._logger.error("{} : Trying update {}.".format(e, symbol))
            flag = -1       # flag for error
        return flag

    def update(self, company_path = ""):
        """
        Construct or update day-1 time-frame database within Yahoo API

        Parameters
        ----------
        `company_path` : ``str``
            The path to the company symbols that must be updated  in
            database. In case the file/path does not exist or is not
            informed use default list `BIG_COMPANIES`.

        """
        self._logger.info("================== FULL UPDATE REQUESTED")
        if (len(company_path) > 0):
            try:
                file_symbols = list(pd.read_csv(company_path)["symbols"])
            except FileNotFoundError as e:
                print(e,"Using list of biggest companies in IBOV index.")
                file_symbols = BIG_COMPANIES
        else:
            print("Empty path to file with company symbols. "
                    "Using biggest companies in IBOV index.")
            file_symbols = BIG_COMPANIES
        nsymbols = len(file_symbols)
        i = 1
        new = 0
        errors = 0
        updated = 0
        non_updated = 0
        for symbol in file_symbols:
            print("[{:2d}/{}]".format(i, nsymbols), end=" ")
            flag = self.update_symbol(symbol)
            if flag == 2:
                new += 1
                print("{} new share introduced".format(symbol))
            elif flag == 1:
                updated += 1
                print("{} updated".format(symbol))
            elif flag == 0:
                non_updated += 1
                print("{} skipped - less than 3 days of delay".format(symbol))
            else:
                errors += 1
                print("{} ! FAILED - check yahoo.log file".format(symbol))
            i += 1
        result_msg = (
            "\nData is up to date {}\n"
            "From {:2d} symbols requested:\n"
            "\t{:2d} new introduced;\n"
            "\t{:2d} updated;\n"
            "\t{:2d} skipped updates\n"
            "\t{:2d} errors.\n"
            "========================================"
            "========================================".format(
            dt.date.today(), nsymbols, new, updated, non_updated, errors)
        )
        self._logger.info(result_msg)
        print(result_msg)

    def __del__(self):
        self._conn.close()



class MetaTrader:
    """
    Deal with CSV files downloaded from MetaTrader.

    """

    def __init__(self):
        os.makedirs("logs/", exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            "logs/mtrader.log", maxBytes=200 * 1024 * 1024, backupCount=1
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        self._logger = logging.getLogger("Logger")
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(handler)

    def get_raw_name_initial_time(self, file_path):
        name = os.path.basename(file_path)
        return name.split("_")[-2]

    def get_raw_name_final_time(self, file_path):
        name = os.path.basename(file_path)
        return name.split("_")[-1].split(".")[0]

    def parse_raw_time(self, raw_time_str):
        """Return `datetime` object based on `raw_time_str`

        Parameters
        ----------
        raw_time_str : ``str``
            Datetime string in format `YYYYmmddHHMM

        Returns
        -------
        raw_time_str : ``datetime.datetime``.
        """
        time_str = (
            raw_time_str[:4]
            + "."
            + raw_time_str[4:6]
            + "."
            + raw_time_str[6:8]
            + " "
            + raw_time_str[8:10]
            + ":"
            + raw_time_str[10:]
            + ":00"
        )
        return dt.datetime.strptime(time_str, "%Y.%m.%d %H:%M:%S")

    def get_period_from_name(self, file_path):
        """Return an temporal interval based on a file path

        Parameters
        ----------
        file_path : ``str``
            A path to the csv file with the meta trader share data. The
            following format is required ``*._YYmmddHHMM_YYmmddHHMM.csv``

        Return
        ------
        interval : ``Dict[str, datetime.datetime]``

        """
        raw_time_str_initial = self.get_raw_name_initial_time(file_path)
        raw_time_str_final = self.get_raw_name_final_time(file_path)
        initial = self.parse_raw_time(raw_time_str_initial)
        final = self.parse_raw_time(raw_time_str_final)
        return {"initial": initial, "final": final}

    def check_disjunction(self, interval_a, interval_b):
        """
        Check if the intervals a and b are disjoint.

        """
        if (
            interval_a["final"] < interval_b["initial"]
            or interval_a["initial"] > interval_b["final"]
        ):
            self._logger.error("Disjoint intervals")
            raise ValueError("Disjoint intervals")

    def create_data_frame(self, file_path, sep=None):
        """Create DataFrame using csv file in `file_path`. The miliseconds are
        ignored during the processing of times.

        Parameters
        ----------
        file_path : ``str``
            A path to the csv file with the meta trader share data.

        Return
        ------
        df : ``pandas.DataFrame``

        """
        df = pd.read_csv(file_path, sep=sep)
        df["<TIME>"] = df[["<TIME>"]].applymap(lambda x: x.split(".")[0])
        return df

    def get_datetime_series(self, df):
        """Create Series concatenating date and time columns of `df`
        ignored during the processing of times.

        Parameters
        ----------
        df : ``pandas.DataFrame``
            A path to the csv file with the meta trader share data.

        Return
        ------
        s : ``pandas.Series``

        """
        s = df["<DATE>"] + " " + df["<TIME>"]
        s = s.map(lambda x: dt.datetime.strptime(x, "%Y.%m.%d %H:%M:%S"))
        return s

    def get_overlap_idx(self, df_a, df_b):
        """Get the first index of df_b in which the datetime is not covered by
        df_a.

        Parameters
        ----------
        df_a and df_b : ``pandas.DataFrame``

        Return
        ------
        index : ``Any``
            A value associated with df_b index.

        Raises
        ------
        ValueError
            If nothing index are found.

        """
        s_a = self.get_datetime_series(df_a)
        s_b = self.get_datetime_series(df_b)
        indices = s_b[s_b > s_a[len(s_a) - 1]].index
        if len(indices) == 0:
            self._logger.error("The is not any ovelap between intervals.")
            raise ValueError("The is not any ovelap between intervals.")
        return indices[0]

    def apply_merge(self, symbol, file_path_a, file_path_b):
        """
        Create a new csv file with the merge of data in the files `file_path_a`
        and `file_path_b`. Non-empty intersection and
        `interval_a["initial"] < interval_b["initial"]` are required.

        """
        period_str = (
            self.get_raw_name_initial_time(file_path_a)
            + "_"
            + self.get_raw_name_initial_time(file_path_b)
        )
        merged_file_path = os.path.join(
            "merged_csv", "{}_M1_{}".format(symbol, period_str)
        )
        df_a = self.share_data_frame(file_path_a)
        df_b = self.share_data_frame(file_path_b)
        idx = self.get_overlap_idx(df_a, df_b)

        merged_df = df_a.append(df_b.loc[idx:], ignore_index=True, sort=False)
        merged_df.to_csv(merged_file_path, index=False)
        return merged_file_path

    def resolve_overlap_and_merge(
        self, symbol, interval_a, interval_b, file_path_a, file_path_b
    ):
        """
        Merge of csv files `file_path_a` and `file_path_b` considering
        the possible data overlaps.

        """
        if interval_a["initial"] < interval_b["initial"]:
            if interval_a["final"] > interval_b["final"]:
                return file_path_a
            return self.apply_merge(symbol, file_path_a, file_path_b)
        else:
            if interval_b["final"] > interval_a["final"]:
                return file_path_b
            return self.apply_merge(symbol, file_path_b, file_path_a)

    def merge_files(self, file_path_a, file_path_b):
        """
        Merge two csv files with overlap data in the period.

        """
        symbol = file_path_a.split("_")[0]
        interval_a = self.get_interval_from_name(file_path_a)
        interval_b = self.get_interval_from_name(file_path_b)
        self.check_disjunction(interval_a, interval_b)
        return self.resolve_overlap_and_merge(
            symbol, interval_a, interval_b, file_path_a, file_path_b
        )

    def parse_csv_files(self, dir_csv_path):
        """Perse csv files from MetaTrader.

        When manually donwloaded, the csv files from MetaTrader may
        become obsolete. Since the new csv files are simply added with
        remaining files to present day. This function merge files in a
        new csv with  a  continuous time ordering. It also informe
        if there are time-gaps among csv files

        Parameters
        ----------
        dir_csv_path : ``str``
            A path to the directory containing the csv files for the meta
            trader share data. The file names following this format
            ``(*.)_*._YYmmddHHMM_YYmmddHHMM.csv``, in which the first group
            is the symbol (or company name).

        """
        if not os.path.exists(dir_csv_path):
            msg = "The path {} does not exist".format(dir_csv_path)
            self._logger.error(msg)
            raise IOError(msg)

        csv_paths = [
            name
            for name in os.listdir(dir_csv_path)
            if name.split(".")[-1] == "csv"
        ]
        symbols = [path.split("_")[0] for path in csv_paths]
        if len(symbols) == 0:
            self._logger.info(
                "There is not any file in {} with expected format".format(
                    dir_csv_path
                )
            )
            return None

        while len(symbols) > 0:
            symbol = symbols[0]
            symbols.remove(symbol)
            file_path_a = csv_paths[0]
            csv_paths.remove(file_path_a)
            if symbol in symbols:
                i = symbols.index(symbol)
                file_path_b = csv_paths[i]
                try:
                    output_file_path = self.merge_files(
                        file_path_a, file_path_b
                    )
                    if output_file_path == file_path_a:
                        os.remove(file_path_b)
                    elif output_file_path == file_path_b:
                        os.remove(file_path_a)
                    else:
                        os.remove(file_path_a)
                        os.remove(file_path_b)
                    csv_paths[i] = output_file_path
                except Exception as e:
                    raise e

    def parse_columns(df):
        """
        Drop `<SPREAD>`, convert columns `<DATE>` and `<TIME>` to the index
        composed of timestamp, and remove brackes from names of remaining
        columns.

        """
        datetime_index = [
            dt.datetime.strptime(
                "{} {}".format(df["<DATE>"][i], df["<TIME>"][i]),
                "%Y.%m.%d %H:%M:%S",
            )
            for i in df.index
        ]
        df.set_index(pd.DatetimeIndex(datetime_index), inplace=True)
        df.index.name = "DateTime"
        df.drop(["<DATE>", "<TIME>", "<SPREAD>"], axis=1, inplace=True)
        columns_rename = {
            "<OPEN>": "Open",
            "<HIGH>": "High",
            "<LOW>": "Low",
            "<CLOSE>": "Close",
            "<VOL>": "Volume",
            "<TICKVOL>": "TickVol",
        }
        df.rename(columns=columns_rename, inplace=True)
        return df

    def create_metratrader_m1_db(self, db_path, dir_csv_path):
        """Create a Sqlite database composed of data from CSV files downloaded
        from MetaTrader.

        Each company must have only one csv file in the path `dir_csv_path`. If
        there  are more than one, the files will be merged applying method
        `parse_csv_files`. If `db_path` already exists, a new file will be
        created.

        Parameters
        ----------
        db_path : ``str``
            The path to the dump file for a Sqlite database.
        dir_csv_path: ``str``
            The path to the directory with CSV files for MetaTrader

        """
        if not os.path.exists(dir_csv_path):
            msg = "The path {} does not exist in this computer".format(
                dir_csv_path
            )
            self._logger.error("{}".format(msg))
            raise IOError(msg)
        self.parse_csv_files(dir_csv_path)
        csv_name_list = [
            name
            for name in os.listdir(dir_csv_path)
            if name.split(".")[-1] == "csv"
        ]
        num_files = len(csv_name_list)
        if num_files == 0:
            msg = "There are no csv files in {}".format(dir_csv_path)
            self._logger.error("{}".format(msg))
            raise IOError(msg)
        if os.path.isfile(db_path):
            os.rename(db_path, db_path + ".{}.old".fromat(dt.datetime.now()))
        dir_name = os.path.dirname(db_path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(db_path)
        for csv_name in csv_name_list:
            symbol = csv_name.split("_")[0]
            df = self.parse_columns(
                pd.read_csv(dir_csv_path + csv_name, sep=None)
            )
            df.to_sql(symbol, con=conn)
        conn.close()
        self._logger.info("Process finished. DB-connection closed\n")
