import os
import sys
import sqlite3
import logging
import logging.handlers
import socket

import pandas as pd
import datetime as dt
import pandas_datareader as pdr

from pathlib import Path

__all__ = ["Yahoo", "MetaTrader", "SocketServer"]

# global variables to automate function calls

DEFT_PORT = 9090
DEFT_ADDRESS = "127.0.0.1"
DEFT_DB_PATH = str(Path.home()) + "/FintelligenceData/"
DEFT_SYMBOLS_PATH = DEFT_DB_PATH + "CompanySymbols_list.txt"
DEFT_CSV_DIR_PATH = DEFT_DB_PATH + "csv_files/"
DEFT_D1_DB_PATH = DEFT_DB_PATH + "Yahoo_D1.db"
DEFT_M1_DB_PATH = DEFT_DB_PATH + "MetaTrader_M1.db"
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

    def __init__(self, db_path = DEFT_D1_DB_PATH):
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
    Creation and update of 1-minute time frame database. This class
    provide an interface to read data in csv files exported from
    MetaTrader and also communicate using a socket connection.

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
        self._default_init_date = dt.datetime(2015,1,2)

    def __get_filename_initial_time(self, file_name):
        return file_name.split("_")[-2]

    def __get_filename_final_time(self, file_name):
        return file_name.split("_")[-1].split(".")[0]

    def __get_filename_symbol(self, file_name):
        return file_name.split("_")[0]

    def __parse_time(self, raw_time_str):
        """
        Return `datetime` object corresponding to `raw_time_str`

        Parameters
        ----------
        raw_time_str : ``str`` Datetime string in format `YYYYmmddHHMM

        Return
        ------
        ``datetime.datetime`` object.
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
        """
        Return an temporal interval based on a file path

        Parameters
        ----------
        file_path : ``str``
            A path to the CSV file with the meta trader share data. The
            following format is required ``*_YYmmddHHMM_YYmmddHHMM.csv``
            where * stands for the share symbol

        Return
        ------
        interval : ``Dict[str, datetime.datetime]``

        """
        file_name = os.path.basename(file_path)
        raw_time_str_initial = self.__get_filename_initial_time(file_name)
        raw_time_str_final = self.__get_filename_final_time(file_name)
        initial = self.__parse_time(raw_time_str_initial)
        final = self.__parse_time(raw_time_str_final)
        return {"initial": initial, "final": final}

    def csv_dataframe_parser(self, file_path, sep=None):
        """
        Create DataFrame using CSV file in `file_path`.
        Remove miliseconds in time column if present.

        Parameters
        ----------
        file_path : ``str`` A path to the CSV file exported from MetaTrader.

        Return
        ------
        df : ``pandas.DataFrame``

        """
        df = pd.read_csv(file_path, sep=sep, engine='python')
        df["<TIME>"] = df[["<TIME>"]].applymap(lambda x: x.split(".")[0])
        return df

    def __get_datetime_series(self, raw_df):
        """
        Create Series concatenating date and time columns of `df`
        ignored during the processing of times.

        Parameters
        ----------
        raw_df : ``pandas.DataFrame`` A raw dataframe obtained from csv parser.

        Return
        ------
        s : ``pandas.Series``

        """
        s = raw_df["<DATE>"] + " " + raw_df["<TIME>"]
        s = s.map(lambda x: dt.datetime.strptime(x, "%Y.%m.%d %H:%M:%S"))
        return s

    def refine_columns(self, df):
        """
        Drop `<SPREAD>`, convert columns `<DATE>` and `<TIME>` to the index
        composed of timestamp, and remove brackes from names of remaining
        columns.

        Parameters
        ----------
        df : ``pandas.DataFrame`` A raw dataframe obtained from csv parser.

        Return
        ------
        df : ``pandas.DataFrame`` With the labels conveniently changed

        """
        datetime_index = self.__get_datetime_series(df)
        df.set_index(datetime_index, inplace=True)
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

    def csv_to_dataframe(self, csv_file_path):
        """From a csv file path return a (refined)dataframe"""
        return self.refine_columns(self.csv_dataframe_parser(csv_file_path))

    def create_new_m1_db(self, db_path = DEFT_M1_DB_PATH,
            dir_csv_path = DEFT_CSV_DIR_PATH):
        """
        Create a Sqlite database from CSV files downloaded from MetaTrader.
        Each company must have only one csv file in the path `dir_csv_path`
        If `db_path` already exists, a new file will be created and the old
        database file will be renamed by appending the suffix with today's
        date and '.old'

        Parameters
        ----------
        db_path : ``str``
            The path to the dump file for a Sqlite database.
        dir_csv_path: ``str``
            The path to the directory with MetaTrader CSV files

        """
        self._logger.info(
                "======================================="
                "======================================="
                "New database from csv files requested\n"
        )
        if not os.path.exists(dir_csv_path):
            msg = "The path {} does not exist in this computer".format(
                dir_csv_path
            )
            self._logger.error("{}".format(msg))
            raise IOError(msg)
        if (dir_csv_path[-1] != "/"):
            dir_csv_path = dir_csv_path + "/"
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
            os.rename(db_path, db_path + ".{}.old".format(dt.datetime.now()))
        dir_name = os.path.dirname(db_path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(db_path)
        i = 1
        for csv_name in csv_name_list:
            symbol = self.__get_filename_symbol(csv_name)
            period = self.get_period_from_name(dir_csv_path + csv_name)
            df = self.csv_to_dataframe(dir_csv_path + csv_name)
            df.to_sql(symbol, con=conn)
            print("[{:2d}/{}] {} from {} to {} introduced".format(
                i, num_files, symbol, period["initial"], period["final"])
            )
            i += 1
        conn.close()
        final_msg = (
            "\nDB construction Process finished. "
            "{} shares introduced from csv files\n".format(num_files)
        )
        print(final_msg)
        self._logger.info(final_msg)

    def update_m1_db(self,
            db_path = DEFT_M1_DB_PATH,
            sym_path = DEFT_SYMBOLS_PATH,
            optional_csv_dir = DEFT_CSV_DIR_PATH):
        """
        Update or create (case it does not exist) the 1-minute time  frame
        database. A socket connection is required to transfer data between
        MetaTrader (client) and python (server), thus it requires the user
        to also run a script (expert advisor) in MetaTrader where the user
        also has to provide an account to properly access the share prices

        Expert Advisor in MetaTrader : `SendListOfSymbols`

        Execute the EA when a message informs it is waiting for the client

        Parameters
        ----------
        `db_path` : ``str``
            full path to database file to be updated. If no one is found
            create it trying to use csv files in `optional_csv_dir`
        `sym_path` : ``str``
            full path to text file with list of symbols to be updated.
            They do not need to match exactly the ones already in the
            database file being updated and new ones can be included.
            If the file is not provided use a default list of biggest
            companies in IBOV index
        `optional_csv_dir` : ``str``
            full path to a directory to use in case the database file
            is not found and must be created. If no CSV files are found
            use directly the socket connection to retrieve data.

        """
        print("\nBuilding/Updating MetaTrader Minute-1 database")
        if len(db_path) == 0:
            self._logger.error("1-minute db path must be non-empty string")
            raise IOError("1-minute db path must be non-empty string")
        dir_name = os.path.dirname(db_path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)
        if (not os.path.isfile(db_path)):
            warn_msg = ("No db file found to update. Building it from "
                    "scratch. Trying to use CSV files ... ")
            try:
                self.create_new_m1_db(db_path,optional_csv_dir)
                warn_msg += "Using CSV files from {}".format(optional_csv_dir)
            except IOError:
                warn_msg += "No CSV files found "
                pass
            print(warn_msg)
            self._logger.warn(warn_msg)
        try:
            sym_file = open(sym_path, "r")
            symbols = [symbol.strip() for symbol in sym_file.readlines()]
            sym_file.close()
        except FileNotFoundError:
            warn_msg = ("\n! WARNING: CompanySymbols_list.txt file not found."
                    " Using default list of important companies of IBOVESPA")
            print(warn_msg)
            self._logger.warn(warn_msg)
            symbols = BIG_COMPANIES
        nsymbols = len(symbols)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' "
                       "ORDER BY name"
        )
        db_symbols = [tabname[0] for tabname in cursor.fetchall()]
        input_dict = dict()
        final_date = dt.datetime.today()
        for symbol in symbols:
            if (symbol in db_symbols):
                cursor = conn.cursor()
                cursor.execute("SELECT DateTime FROM {} ORDER BY DateTime "
                        "DESC LIMIT 1".format(symbol)
                )
                date_str = cursor.fetchall()[0][0]
                last_update = dt.datetime.strptime(date_str,
                        "%Y-%m-%d %H:%M:%S"
                )
                start_date = last_update + dt.timedelta(minutes=1)
            else:
                start_date = self._default_init_date
            date_diff = final_date - start_date
            if (date_diff > dt.timedelta(days = 1)):
                input_dict[symbol] = (start_date,final_date)
        mt_server = SocketServer()
        data = mt_server.get_m1_ohlc_dataset(input_dict)
        for symbol in data.keys():
            if type(data[symbol]) is str: continue
            data[symbol].to_sql(symbol, con=conn, if_exists = "append")
        del mt_server
        conn.close()

        symbols_failed = [
                sym
                for sym in data.keys()
                if type(data[sym]) is str
        ]
        symbols_ignored = [
                sym
                for sym in symbols
                if sym not in input_dict.keys()
        ]
        new_symbols = [
                sym
                for sym in symbols 
                if (sym not in db_symbols and sym not in symbols_failed)
        ]
        err_count = len(symbols_failed)
        suc_count = len(symbols) - len(symbols_ignored) - len(symbols_failed)
        new_count = len(new_symbols)
        skp_count = len(symbols_ignored)
        upd_count = suc_count - new_count
        if (err_count > 0):
            final_err_msg = ("\nFAILURE FETCHING THE FOLLOWING TICKERS "
                    "IN DATE {}".format(dt.datetime.now())
            )
            for symbol in symbols_failed:
                final_err_msg += ("\n{} {}".format(symbol,data[symbol]))
            self._logger.error(final_err_msg)
        final_info_msg = (
                "\nFrom {} symbols requested\n"
                "\t{} introduced\n"
                "\t{} needed update\n"
                "\t{} already up to date\n"
                "\tand {} failed\n".format(
                len(symbols), new_count, upd_count, skp_count, err_count)
        )
        self._logger.info(final_info_msg)
        print(final_info_msg)
        print("db-connection closed. Update finished.")



class SocketServer:
    """
    Class to provide an interface to communicate directly with MetaTrader.
    It create a server in python and accept connections from MetaTrader
    that is referred as client. The methods must be used together with the
    corresponding expert advisors in MetaTrader where an account must be
    provided to be able to access data.
    """
    def __init__(self, address = DEFT_ADDRESS, port = DEFT_PORT,
                 bytes_lim = 8192, listen = True):
        self.sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 'setsockopt' to avoid openned socket in TIME_WAIT state
        self.sck.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.today = dt.datetime.today()
        self.port = port
        self.address = address
        self.bytes_lim = bytes_lim  # max bytes in a single receive
        self.sck.bind((self.address, self.port))
        self.isListening = False    # socket server listening status
        self.conn = None            # current openned connection service
        self.addr = None            # address of current connection
        if (listen): self.startListening()

    def startListening(self):
        self.sck.listen(1)
        self.isListening = True

    def stopListening(self):
        try: self.sck.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        self.isListening = False

    def refresh(self):
        """
        Refresh server by shuting down and then setting it to listening
        again. This is useful to clean any bytes stucked in buffer when
        some error occurred in previous communications
        """
        self.stopListening()
        self.startListening()

    def __communicate(self, message):
        """Send a `message` to MetaTrader Client and return the response"""
        if (self.conn == None):
            raise AttributeError("No openned connection in __communicate")
        if (not self.isListening):
            raise AttributeError("Server is not listening in __communicate")
        self.conn.send(bytes(message,"utf-8"))
        raw_data = self.conn.recv(self.bytes_lim)
        return raw_data.decode("utf-8")

    def __assertSymbol(self, symbol):
        """Return True if company `symbol` exists in MetaTrader"""
        if (self.__communicate(symbol).upper == "ERROR"): return False
        return True

    def get_raw_m1(self, symbol, initial_date, final_date, it = False):
        """
        Receive raw OHLC data from MetaTrader in a list of strings with
        each element corresponding to a day

        Parameters
        ----------
        `symbol` : ``str``
            company share code, ex: "PETR4"
        `initial_date` : ``datetime.datetime``
            Initial date and time to start retrieving data
        `final_date` : ``datetime.datetime``
            final date to stop retrieving data
        `it` : ``bool``
            Indicate if this function is iteratively called in which
            case supress information printing

        Return
        ------
        `str_fulldata_list` : ``list``
            Each element in the list has 1 day of OHLC data in string
            format, separated by commas (each minute) and spaces (the
            data OHLC fields)

        """
        if (type(symbol) != str):
            raise TypeError("Symbol must be a string")
        if (type(initial_date) == type(final_date) == dt.datetime):
            str_dt1 = initial_date.strftime("%Y.%m.%d %H:%M")
            str_dt2 = final_date.strftime("%Y.%m.%d %H:%M")
        else:
            raise TypeError("Initial and final date must be datetime objects.")
        if (final_date < initial_date):
            raise ValueError("Final date must be forward of initial one.")
        if (not it):
            print("\nWaiting for client (MetaTrader) connection ...",end=" ")
            sys.stdout.flush()
        # Wait connection from Client (MetaTrader)
        if (not self.isListening):
            self.startListening()
        self.conn, self.addr = self.sck.accept()
        if (not it):
            print("Connected to", self.addr)
        if (not self.__assertSymbol(symbol)):
            self.conn.close()
            raise ValueError("Some problem occurred fetching {} in "
            "MetaTrader. Assert it exists".format(symbol))
        str_fulldata_list = []
        str_data = ""
        # inform initial and final dates to MetaTrader
        # in a string separated by a underscore
        self.conn.send(bytes(str_dt1 + "_" + str_dt2,"utf-8"))
        if (not it):
            print("Transfering {} data through socket connection ...".format(
            symbol), end=" ")
            sys.stdout.flush()
        while True:
            raw_data = self.conn.recv(self.bytes_lim)
            str_data += raw_data.decode("utf-8")
            if not raw_data: break
            # Underscore character _ separate data(string) set in days
            if "_" in str_data:
                str_split = str_data.split("_")
                # split days as list elements
                for part in str_split[:len(str_split)-1]:
                    if len(part) > 0: str_fulldata_list.append(part)
                str_data = str_split[-1]
        if (not it): print("Done\n")
        self.conn.close()
        self.conn = None
        self.addr = None
        # Append remaining data after last receipt(break in while)
        if (len(str_data) > 0): str_fulldata_list.append(str_data)
        return str_fulldata_list

    def get_m1_ohlc_dataframe(self, symbol, initial_date,
                                final_date, it = False):
        """
        Retrieve share data from MetaTrader in 1-minute time frame.

        Parameters
        ----------
        `symbol` : ``str``
            company share code, ex: "PETR4"
        `initial_date` : ``datetime.datetime``
            Initial date and time to start retrieving data
        `final_date` : ``datetime.datetime``
            final date to stop retrieving data
        `it` : ``bool``
            Indicate if this function is iteratively called in which
            case supress information printing

        Return
        ------
        `df` : ``pandas.DataFrame``

        """
        str_list = self.get_raw_m1(symbol,initial_date,final_date,it)
        ohlc_list = []  # core data of pandas data-frame
        date_list = []  # indexing of pandas data-frame
        for day_data in str_list:
            for ohlc_str in day_data.split(","):
                ohlc_split = ohlc_str.split(" ")
                date_str = ohlc_split[0] + " " + ohlc_split[1]
                ohlc_list.append([float(x) for x in ohlc_split[2:]])
                date_list.append(pd.Timestamp(date_str))
        df = pd.DataFrame(
                ohlc_list,index=date_list,
                columns=["Open","High","Low","Close","TickVol","Volume"]
        )
        df["TickVol"] = df["TickVol"].astype(int)
        df["Volume"] = df["Volume"].astype(int)
        df.index.name = "DateTime"
        return df

    def get_m1_ohlc_dataset(self, symbols_dict):
        """
        Retrieve data for a set of shares

        Parameters
        ----------
        `symbols_dict` : ``dic[symbol : (initial_date, final_date)]``
            provide the shares symbols as keys and values with the
            corresponding period as a tuple of 2 ``datetime.datetime``

        Return
        ------
        `data_dict` : ``pandas.DataFrame``
            Use the shares symbols as keys and the values as dataframe
            with Open-High-Low-Close-TickVol-Volume data per minute

        """
        AllSymbols_str = ""
        if len(symbols_dict.keys()) == 0: return dict()
        for symbol in symbols_dict.keys():
            if (type(symbol) != str):
                raise KeyError("Non string-type symbol found in "
                               "input dictionary key: {}".format(symbol))
            AllSymbols_str += "_" + symbol
        AllSymbols_str = AllSymbols_str[1:]
        if (len(AllSymbols_str) == 0):
            raise ValueError("No symbols found in input symbol list")
        print("\nWaiting for client (MetaTrader) connection ...", end = " ")
        sys.stdout.flush()
        symbols_requested = AllSymbols_str.split("_")
        if (not self.isListening): self.startListening()
        self.conn, self.addr = self.sck.accept()
        # Send in message the list of requested symbols to MetaTrader
        # and receive only the valid symbols as response
        valid_symbols_str = self.__communicate(AllSymbols_str)
        valid_symbols = valid_symbols_str.split("_")
        self.conn.close()
        print("Connected to", self.addr)

        Nsymbols_requested = len(symbols_requested)
        Nsymbols_valid = len(valid_symbols)
        print("\n{} valid symbols of {} to be updated/created".format(
               Nsymbols_valid,Nsymbols_requested))
        data_dict = dict()
        i = 0
        print("Working in ...\n")
        for symbol in symbols_requested:
            i = i + 1
            initial_date = symbols_dict[symbol][0]
            final_date = symbols_dict[symbol][1]
            print("[{:2d}/{}] {}".format(i,Nsymbols_requested,symbol),end=" ")
            if (symbol not in valid_symbols):
                err_msg = "Symbol not found in MetaTrader"
                print("! ERROR.",err_msg)
                data_dict[symbol] = "ERROR: " + err_msg
                continue
            if final_date < initial_date:
                err_msg = ("Invalid initial and final dates"
                " from {} to {}".format(initial_date,final_date))
                print("! ERROR.",err_msg)
                data_dict[symbol] = "ERROR: " + err_msg
                continue
            print("from",initial_date,"to",final_date)
            sys.stdout.flush()
            data_dict[symbol] = self.get_m1_ohlc_dataframe(symbol,
                                initial_date,final_date,True)
        return data_dict

    def __del__(self):
        self.sck.close()
