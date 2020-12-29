import datetime as dt
import os
import logging
import sqlite3

import pandas as pd
import pandas_datareader as pdr
from pathlib import Path


DEFAULT_DB_PATH = str(Path.home()) + "/FintelligenceData/"
CSV_FILES_PATH = str(Path.home()) + "/FintelligenceData/csv_files/"
INITIAL_DATE_D1 = dt.date(2010, 1, 2)
FINAL_DATE_D1 = dt.date.today() - dt.timedelta(days=1)


class Yahoo:
    """Define a database in Sqlite using data from Yahoo databases for
    collected shared prices.

    For now, this implementation analysis the day-1 frequency of Yahoo data.

    Parameters
    ----------
    db_path : ``str``
        The path to the dump file for a Sqlite database.

    """

    def __init__(self, db_path):
        handler = logging.handlers.RotatingFileHandler(
            "yahoo.log", maxBytes=200 * 1024 * 1024, backupCount=1
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

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._cursor = self._conn.cursor()

    def get_symbols(self):
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
        Get date associated with `symbol` plus one day

        """
        full_date = self._cursor.execute(
            "SELECT Date FROM {} ORDER BY Date DESC LIMIT 1".format(symbol)
        ).fetchall()[0][0]
        date_str = full_date.split(" ")[0]
        last_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        init_day1 = last_date + dt.timedelta(days=1)
        return init_day1

    def update_symbol(self, init_day1, symbol, final_day1, if_exist="fail"):
        """
        Update data for the `symbol` in the interval
        [`init_day1`, `final_day`].

        """
        try:
            df = pdr.DataReader(symbol + ".SA", "yahoo", init_day1, final_day1)
            df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)
            df.to_sql(symbol, con=self._conn, if_exists=if_exist)
            self._logger.info("Share {} successfully updated.".format(symbol))
        except Exception as e:
            self._logger.error(
                "{} : During the fetch for symbol {}.".format(e, symbol)
            )
            return -1
        else:
            return 1

    def update_known_symbol(self, init_day1, symbol, final_day1):
        """
        Update data for the `symbol` that already is in the connected database.

        """
        if init_day1 + dt.timedelta(days=3) >= final_day1:
            print(
                "Share {} will be not updated,"
                " less than 3 days of delay.".format(symbol)
            )
            return 0
        else:
            return self.update_symbol(init_day1, symbol, final_day1, "append")

    def update(self, company_path):
        """Construct or update day-1 frequency for Yahoo database

        Parameters
        ----------
        company_path : ``str``
            The path to the company symbols associated with Yahoo database

        """
        try:
            file_symbols = list(pd.read_csv(company_path)["symbols"])
        except FileNotFoundError as e:
            print(e)
            exit(1)
        db_symbols = self.get_symbols()

        errors = 0
        updated = 0
        non_updated = 0
        init_day1 = self.get_date()
        final_day1 = dt.date.today() - dt.timedelta(days=1)
        for symbol in file_symbols:
            if symbol in db_symbols:
                flag = self.update_known_symbol(symbol, final_day1)
            else:
                flag = self.update_symbol(init_day1, symbol, final_day1)

            if flag == 1:
                updated += 1
            elif flag == 0:
                non_updated += 1
            else:
                errors += 1
        self._logger.info(
            "Data updated for {}"
            " with {} successful updates, "
            "{} skipped updates, and {} errors.".format(
                updated, non_updated, errors, dt.datetime.now()
            )
        )

    def __del__(self):
        self._conn.close()


def getCSV_period(filename):
    """ Retrieve period(dictionary) from filename in MetaTrader convention """
    raw_str = filename.split("_")[2]
    initial_str = (
        raw_str[:4]
        + "."
        + raw_str[4:6]
        + "."
        + raw_str[6:8]
        + " "
        + raw_str[8:10]
        + ":"
        + raw_str[10:]
        + ":00"
    )
    raw_str = filename.split("_")[3].split(".")[0]
    final_str = (
        raw_str[:4]
        + "."
        + raw_str[4:6]
        + "."
        + raw_str[6:8]
        + " "
        + raw_str[8:10]
        + ":"
        + raw_str[10:]
        + ":00"
    )
    initial = dt.datetime.strptime(initial_str, "%Y.%m.%d %H:%M:%S")
    final = dt.datetime.strptime(final_str, "%Y.%m.%d %H:%M:%S")
    return {"initial": initial, "final": final}


def mergeFiles(filenameA, filenameB):
    """ Merge two csv files that have an overlapping data period """
    symbol = filenameA.split("_")[0]  # symbol common to both files
    periodA = getCSV_period(
        filenameA
    )  # get period as dict. of datetime data-types
    periodB = getCSV_period(filenameB)
    # check if there is an ovelap in the periods. Raise error if the intervals are disjoint
    if (
        periodA["final"] < periodB["initial"]
        or periodA["initial"] > periodB["final"]
    ):
        raise ValueError("Disjoint Intervals")
    if periodA["initial"] < periodB["initial"]:
        # file A cover a former initial date
        if periodA["final"] > periodB["final"]:
            # file B is contained in file A - nothing to merge
            return filenameA
        # must merge files since B final date/time is ahead of A
        # Format string to set period in filename of resulting merged data
        period_str = filenameA.split("_")[2] + "_" + filenameB.split("_")[3]
        merged_filename = "{}_M1_{}".format(symbol, period_str)
        # Load data from files
        dfA = pd.read_csv(CSV_FILES_PATH + filenameA, sep="\t")
        dfB = pd.read_csv(CSV_FILES_PATH + filenameB, sep="\t")
        # Take start index in B corresponging to the A final date/time
        A_last_ind = len(dfA.index) - 1
        B_start_ind = (
            dfB.loc[
                (dfA["<DATE>"][A_last_ind] == dfB["<DATE>"])
                & (dfA["<TIME>"][A_last_ind] == dfB["<TIME>"])
            ].index[0]
            + 1
        )
        # properly merge files and save in new csv file
        newdf = dfA.append(
            dfB.iloc[B_start_ind:], ignore_index=True, sort=False
        )
        newdf.to_csv(CSV_FILES_PATH + merged_filename, sep="\t", index=False)
        return merged_filename
    else:
        # file B cover a former initial date - reverse case of the 'if' block above
        # For more information see the comments in the 'if' block above
        if periodB["final"] > periodA["final"]:
            # file A is contained in file B - nothing to merge
            return filenameB
        period_str = filenameB.split("_")[2] + "_" + filenameA.split("_")[3]
        merged_filename = "{}_M1_{}".format(symbol, period_str)
        dfA = pd.read_csv(CSV_FILES_PATH + filenameA, sep="\t")
        dfB = pd.read_csv(CSV_FILES_PATH + filenameB, sep="\t")
        B_last_ind = len(dfB.index) - 1
        A_start_ind = (
            dfA.loc[
                (dfB["<DATE>"][B_last_ind] == dfA["<DATE>"])
                & (dfB["<TIME>"][B_last_ind] == dfA["<TIME>"])
            ].index[0]
            + 1
        )
        newdf = dfB.append(
            dfA.iloc[A_start_ind:], ignore_index=True, sort=False
        )
        newdf.to_csv(CSV_FILES_PATH + merged_filename, sep="\t", index=False)
        return merged_filename


def updateCSVFiles():
    """Auxiliar function to merge csv files donwloaded from MetaTrader
    ===============================================================
    When manually donwloaded, the csv files from MetaTrader may
    become obsolete. Then by simply adding new csv  files  with
    remaining data to present day in  the  folder  'csv_files',
    this function merge files in a new csv  with  a  continuous
    time ordering. It also informe if there are time-gaps among
    csv files
    """
    # DEFAULT WARNING AND ERROR MESSAGES
    path_err_msg = (
        "ERROR : The path {} does not exist in this computer".format(
            CSV_FILES_PATH
        )
    )
    csv_files_err_msg = "ERROR : There are no csv files in {}".format(
        CSV_FILES_PATH
    )
    # TRY TO FIND THE PATH OF CSV DILES
    if not os.path.exists(CSV_FILES_PATH):
        raise IOError(path_err_msg)
    print("\nScanning csv files ...\n")
    # list of strings with all csv file names
    csv_filename_list = [
        name
        for name in os.listdir(CSV_FILES_PATH)
        if name.split(".")[-1] == "csv"
    ]
    symbols = [
        csv_filename.split("_")[0] for csv_filename in csv_filename_list
    ]
    nfiles = len(symbols)
    if nfiles < 1:
        raise IOError(csv_files_err_msg)  # There are no csv files
    while len(symbols) > 0:
        # get first symbol
        symbol = symbols[0]
        symbols.remove(symbol)
        filenameA = csv_filename_list[0]
        csv_filename_list.remove(filenameA)
        # check if there is a repetition (other csv of the same company to update)
        if symbol in symbols:
            print("More than one csv file for {} ... ".format(symbol), end="")
            i = symbols.index(symbol)
            filenameB = csv_filename_list[i]
            # Merge the two csv files
            try:
                output_filename = mergeFiles(filenameA, filenameB)
                # remove files that were merged
                os.remove(CSV_FILES_PATH + filenameA)
                os.remove(CSV_FILES_PATH + filenameB)
                # include new file generated replacing the old one
                csv_filename_list[i] = output_filename
                print("Files merged")
            except ValueError:
                print("Could not merge files - disjoint time intervals")
    print("\nFinished scanning csv files\n")


def refineDF(df):
    """Refine data extracted from csv files using a better convention
    ==============================================================
    Especially the column names comes with <> and date-time are
    given as strings. Here remove <> and set date-time properly
    as python time-stamp. Called in 'createDB_MetaTraderCSV'
    """
    refined = df
    # Date and Time are given in separate rows as the rows labels are just numbers
    # Therefore merge these two rows information to set as labes of the  dataframe
    datetimeIndex = [
        dt.datetime.strptime(
            "{} {}".format(df["<DATE>"][i], df["<TIME>"][i]),
            "%Y.%m.%d %H:%M:%S",
        )
        for i in df.index
    ]
    pandas_datetimeIndex = pd.DatetimeIndex(datetimeIndex)
    refined.set_index(pandas_datetimeIndex, inplace=True)
    refined.index.name = "DateTime"
    refined.drop(
        ["<DATE>", "<TIME>", "<SPREAD>"], axis=1, inplace=True
    )  # no longer needed ?
    # Remove annoyng <> bracket notation
    columns_rename = {
        "<OPEN>": "Open",
        "<HIGH>": "High",
        "<LOW>": "Low",
        "<CLOSE>": "Close",
        "<VOL>": "Volume",
        "<TICKVOL>": "TickVol",
    }
    # columns_rename = {column:column[1:-1] for column in refined.columns if ('<' in column and '>' in column)}
    refined.rename(columns=columns_rename, inplace=True)
    return refined


def createDB_MetaTraderCSV_M1(db_filename="BRSharesMetaTrader_M1.db"):
    """CSV files Downloaded from MetaTrader is stored in a sql database
    ================================================================
    From csv files exported from MetaTrader this function creates
    a sql database. Each company must have ONLY ONE .csv file  in
    the path introduced in CSV_FILES_PATH variable. If there  are
    more than one .csv file per company,  which  were  downloaded
    aiming to update data, try 'updateCSVFiles' to merge them. If
    a database with 'db_filename' already exists raise an error
    """
    print("\nBuilding MetaTrader minute-1 database from CSV files ...\n")
    full_db_path = DEFAULT_DB_PATH + db_filename
    # DEFAULT WARNING AND ERROR MESSAGES
    path_err_msg = (
        "ERROR : The path {} does not exist in this computer".format(
            CSV_FILES_PATH
        )
    )
    exist_err_msg = (
        "ERROR : MetaTrader database file {} already exists".format(
            full_db_path
        )
    )
    csv_files_err_msg = "ERROR : There are no csv files in {}".format(
        CSV_FILES_PATH
    )
    new_msg = "[{:2d},{}] {} introduced in the database"
    # TRY TO FIND THE PATH OF CSV DILES
    if not os.path.exists(CSV_FILES_PATH):
        raise IOError(path_err_msg)
    # Get list of all csv file names
    csv_filename_list = [
        name
        for name in os.listdir(CSV_FILES_PATH)
        if name.split(".")[-1] == "csv"
    ]
    nfiles = len(csv_filename_list)
    if nfiles < 1:
        raise IOError(csv_files_err_msg)  # There are no csv files
    if os.path.isfile(full_db_path):
        raise IOError(exist_err_msg)
    conn = sqlite3.connect(full_db_path)
    symbol_id = 1
    for csv_filename in csv_filename_list:
        symbol = csv_filename.split("_")[
            0
        ]  # Take company symbol from csv file name
        raw_df = pd.read_csv(CSV_FILES_PATH + csv_filename, sep="\t")
        refined_df = refineDF(raw_df)
        refined_df.to_sql(symbol, con=conn)
        print(new_msg.format(symbol_id, nfiles, symbol))
        symbol_id += 1
    conn.close()  # connection closed
    print("\nProcess finished. DB-connection closed\n")


if __name__ == "__main__":
    updateYahooDB_D1()
    createDB_MetaTraderCSV_M1()
