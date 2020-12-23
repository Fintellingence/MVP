import os
import sqlite3
import pandas as pd
import datetime as dt
import pandas_datareader as pdr # needed to read Yahoo data
from pathlib import Path



# GLOBAL VARIABLES
# ----------------
DEFAULT_DB_PATH = str(Path.home())+'/FintelligenceData/'
CSV_FILES_PATH = str(Path.home())+'/FintelligenceData/csv_files/'
INITIAL_DATE_D1 = dt.date(2010,1,2)
FINAL_DATE_D1 = dt.date.today() - dt.timedelta(days=1)
DB_NAME = 'BRShares_Intraday1M.db'  # name of database file



def updateYahooDB_D1():
    """ Function to construct or update Day-1 frequency database using Yahoo
        --------------------------------------------------------------------
        pandas_datareader module provide functions to extract data
        from web and here we use to take shares price from  Yahoo.
        The company symbols to download data must be  given  in  a
        plain text file 'symbols_list_D1.txt'
    """
    # DEFAULT WARNING AND ERROR MESSAGES
    path_err_msg = "ERROR fetching folder 'shares_prices' in BackupHD external Hard disk device."
    fetch_error_msg = "[{:2d}/{}] ! FATAL ERROR on fetching {}. Assert the symbol is correct"
    not_up_msg = "[{:2d}/{}] Share {} has less than 3 days of delay. Not updating it."
    up_msg = "[{:2d},{}] Share {} successfully updated."
    new_msg = "[{:2d},{}] New Share : {} introduced in the database"
    # TRY TO FIND THE PATH AND OPEN THE .DB FILE
    if (not os.path.exists(DEFAULT_DB_PATH)): raise IOError(path_err_msg)
    db_filename = 'BRsharesYahoo_D1.db'
    full_db_path = DEFAULT_DB_PATH + db_filename
    # Tickers/Symbols to be fetched passed by a list of strings from a .txt file
    symbols_list = list(pd.read_csv(DEFAULT_DB_PATH+'symbols_list_D1.txt')['symbols'])
    nsymbols = len(symbols_list) # total number of companies to track
    # OPEN DATABASE
    conn = sqlite3.connect(full_db_path)
    # CHECK LIST OF TICKERS ALREADY INITIALIZED IN THE DATABASE
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    db_symbols_list = []
    # On scanning the cursor from sql database the tables name
    # are the first element of a tuple given as string
    for table in cursor.fetchall(): db_symbols_list.append(table[0])
    # Check if the shares price have already been initialized.
    # In positive case read the last day it was updated and
    # update only in case it has more than 3 days of delay
    symbol_id = 1
    error_count = 0
    symbols_failed = []
    print('\n')
    for symbol in symbols_list:
        if (symbol in db_symbols_list):
            # SHARE ALREADY IN THE DB. MUST CHECK FOR UPDATES
            cursor = conn.cursor()
            cursor.execute("SELECT Date FROM {} ORDER BY Date DESC LIMIT 1".format(symbol))
            full_date = cursor.fetchall()[0][0]
            # The date is recorded with hours and minutes which are useless
            # Then split the data and take only the Year-Month-Day part
            date_str  = full_date.split(' ')[0]
            last_date = dt.datetime.strptime(date_str,'%Y-%m-%d').date()
            init_date = last_date + dt.timedelta(days=1)
            if (init_date + dt.timedelta(days=3) >= FINAL_DATE_D1):
                print(not_up_msg.format(symbol_id,nsymbols,symbol))
            else:
                try:
                    df = pdr.DataReader(symbol+'.SA','yahoo',init_date,default_final_date)
                    df.rename(columns={'Adj Close':'AdjClose'},inplace=True)
                    df.to_sql(symbol,con=conn,if_exists='append')
                    print(up_msg.format(symbol_id,nsymbols,symbol))
                except:
                    print(fetch_error_msg.format(symbol_id,nsymbols,symbol))
                    symbols_failed.append(symbol)
                    error_count = error_count + 1
        else:
            # TICKER NOT INITIALIZED YET. INTRODUCE IT
            try:
                df = pdr.DataReader(symbol+'.SA','yahoo',INITIAL_DATE_D1,FINAL_DATE_D1)
                df.rename(columns={'Adj Close':'AdjClose'},inplace=True)
                df.to_sql(symbol,con=conn)
                print(new_msg.format(symbol_id,nsymbols,symbol))
            except:
                print(fetch_error_msg.format(symbol_id,nsymbols,symbol))
                symbols_failed.append(symbol)
                error_count = error_count + 1
        symbol_id = symbol_id + 1
    # FINISHED UPDATE OF SHARES GIVEN IN THE TEXT FILE - CLOSE DB CONNECTION
    conn.close() # connection closed
    if (error_count > 0):
        with open(DEFAULT_DB_PATH+'Yahoo_database_D1.log','w') as logfile:
            logfile.write("FAILURE FETCHING THE FOLLOWING TICKERS IN DATE {}".format(dt.datetime.now()))
            for symbol in symbols_failed:
                logfile.write("\n{}".format(symbol))
    print('\nData updated with {} errors. Connection closed'.format(error_count))



def getCSV_period(filename):
    # Given the file name as export by MetaTrader extract the corresponding period
    raw_str = filename.split('_')[2]
    initial_str = raw_str[:4]+'.'+raw_str[4:6]+'.'+raw_str[6:8]+' '+raw_str[8:10]+':'+raw_str[10:]+':00'
    raw_str = filename.split('_')[3].split('.')[0]
    final_str = raw_str[:4]+'.'+raw_str[4:6]+'.'+raw_str[6:8]+' '+raw_str[8:10]+':'+raw_str[10:]+':00'
    initial = dt.datetime.strptime(initial_str,'%Y.%m.%d %H:%M:%S')
    final = dt.datetime.strptime(final_str,'%Y.%m.%d %H:%M:%S')
    return {'initial':initial,'final':final}



def mergeFiles(filenameA,filenameB):
    symbol = filenameA.split('_')[0]    # symbol common to both files
    periodA = getCSV_period(filenameA)  # get period as dict. of datetime data types
    periodB = getCSV_period(filenameB)
    # check if there is an ovelap in the periods. Raise error if the intervals are disjoint
    if (periodA['final'] < periodB['initial'] or periodA['initial'] > periodB['final']):
        raise ValueError
    if periodA['initial'] < periodB['initial']:
        # file A cover a former initial date
        if periodA['final'] > periodB['final']:
            # file B is contained in file A - nothing to merge
            return filenameA
        # must merge files since B final date/time is ahead of A
        # Format string to set period in filename of resulting merged data
        period_str = filenameA.split('_')[2]+'_'+filenameB.split('_')[3]
        merged_filename = '{}_M1_{}'.format(symbol,period_str)
        # Load data from files
        dfA = pd.read_csv(CSV_FILES_PATH+filenameA,sep='\t')
        dfB = pd.read_csv(CSV_FILES_PATH+filenameB,sep='\t')
        # Take start index in B corresponging to the A final date/time
        A_last_ind = len(dfA.index)-1
        B_start_ind = dfB.loc[(dfA['<DATE>'][A_last_ind] == dfB['<DATE>']) & (dfA['<TIME>'][A_last_ind] == dfB['<TIME>'])].index[0]+1
        # properly merge files and save in new csv file
        newdf = dfA.append(dfB.iloc[B_start_ind:],ignore_index=True,sort=False)
        newdf.to_csv(CSV_FILES_PATH+merged_filename,sep='\t',index=False)
        return merged_filename
    else:
        # file A cover a former initial date - reverse case of the 'if' block above
        # For more information see the comments in the 'if' block above
        if periodB['final'] > periodA['final']:
            # file A is contained in file B - nothing to merge
            return filenameB
        period_str = filenameB.split('_')[2]+'_'+filenameA.split('_')[3]
        merged_filename = '{}_M1_{}'.format(symbol,period_str)
        dfA = pd.read_csv(CSV_FILES_PATH+filenameA,sep='\t')
        dfB = pd.read_csv(CSV_FILES_PATH+filenameB,sep='\t')
        B_last_ind = len(dfB.index)-1
        A_start_ind = dfA.loc[(dfB['<DATE>'][B_last_ind] == dfA['<DATE>']) & (dfB['<TIME>'][B_last_ind] == dfA['<TIME>'])].index[0]+1
        newdf = dfB.append(dfA.iloc[A_start_ind:],ignore_index=True,sort=False)
        newdf.to_csv(CSV_FILES_PATH+merged_filename,sep='\t',index=False)
        return merged_filename



def updateCSV():
    # DEFAULT WARNING AND ERROR MESSAGES
    path_err_msg = "ERROR : The path {} does not exist in this computer"
    new_msg = "[{},{}] {} introduced in the database"
    # TRY TO FIND THE PATH OF CSV DILES
    if (not os.path.exists(CSV_FILES_PATH)): raise IOError(path_err_msg.format(CSV_FILES_PATH))
    print('\nScanning csv files\n')
    # list of strings with all csv file names
    csv_filename_list = [name for name in os.listdir(CSV_FILES_PATH) if name.split('.')[-1]=='csv']
    symbols = [csv_filename.split('_')[0] for csv_filename in csv_filename_list]
    while len(symbols) > 0:
        # get first symbol
        symbol = symbols[0]
        symbols.remove(symbol)
        filenameA = csv_filename_list[0]
        csv_filename_list.remove(filenameA)
        # check if there is a repetition (new csv of the same asset to update)
        if symbol in symbols:
            print('More than one csv file for {} ... '.format(symbol),end='')
            i = symbols.index(symbol)
            filenameB = csv_filename_list[i]
            # Merge the two csv files
            try:
                output_filename = mergeFiles(filenameA,filenameB)
                # remove files that were merged
                os.remove(CSV_FILES_PATH+filenameA)
                os.remove(CSV_FILES_PATH+filenameB)
                # include new file generated
                csv_filename_list[i] = output_filename
                print('Files merged')
            except ValueError:
                print('Could not merge files data - disjoint time intervals')
    print('\n\nFinished scanning csv files\n')



def refineDF(df):
    """ Refine data extracted from csv files using a better convention
        ==============================================================
        Especially the column names comes with <> and date-time are
        given as strings. Here remove <> and set date-time properly
        as python time-stamp. Called in 'createDB_MetaTraderCSV'
    """
    refined = df
    # Date and Time are given in separate rows as the rows labels are just numbers
    # Therefore merge these two rows information to set as labes of the  dataframe
    datetimeIndex = [dt.datetime.strptime('{} {}'.format(df['<DATE>'][i],df['<TIME>'][i]),'%Y.%m.%d %H:%M:%S') for i in df.index]
    pandas_datetimeIndex = pd.DatetimeIndex(datetimeIndex)
    refined.set_index(pandas_datetimeIndex,inplace=True)
    refined.index.name = 'DateTime'
    refined.drop(['<DATE>','<TIME>','<SPREAD>'],axis=1,inplace=True) # no longer needed ?
    # Remove annoyng <> bracket notation
    columns_rename = {'<OPEN>':'Open','<HIGH>':'High','<LOW>':'Low','<CLOSE>':'Close','<VOL>':'Volume','<TICKVOL>':'TickVol'}
    # columns_rename = {column:column[1:-1] for column in refined.columns if ('<' in column and '>' in column)}
    refined.rename(columns=columns_rename,inplace=True)
    return refined



def createDB_MetaTraderCSV_1M():
    """ Data Downloaded from MetaTrader is stored in a sql database
        ===========================================================
        From csv files
    """
    # DEFAULT WARNING AND ERROR MESSAGES
    path_err_msg = "ERROR : The path {} does not exist in this computer"
    new_msg = "[{:2d},{}] {} introduced in the database"
    # TRY TO FIND THE PATH OF CSV DILES
    if (not os.path.exists(CSV_FILES_PATH)): raise IOError(path_err_msg.format(CSV_FILES_PATH))
    # list of strings with all csv file names
    csv_filename_list = [name for name in os.listdir(CSV_FILES_PATH) if name.split('.')[-1]=='csv']
    nsymbols = len(csv_filename_list)
    conn = sqlite3.connect(DB_NAME)
    symbol_id = 1
    print('\n')
    for csv_filename in csv_filename_list:
        symbol = csv_filename.split('_')[0]
        raw_df = pd.read_csv(CSV_FILES_PATH+csv_filename,sep='\t')
        refined_df = refineDF(raw_df)
        refined_df.to_sql(symbol,con=conn)
        print(new_msg.format(symbol_id,nsymbols,symbol))
        symbol_id += 1
    conn.close() # connection closed
    print('\n\nProcess finished. Connection closed\n')



def loadFromDB(symbol):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql('SELECT * FROM {}'.format(symbol),con=conn)
    df.index = pd.to_datetime(df['DateTime'])
    df.drop(['DateTime'],axis=1,inplace=True) # no longer needed
    conn.close()
    return df



# updateCSV()
# createDB_MetaTraderCSV()
