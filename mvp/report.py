import pandas as pd
import mvp

def trade_book(refined_data, primary_model, operation_parameters):
    label_data = mvp.labels.Labels(primary_model.events,refined_data.df[['Close']],operation_parameters).label_data
    entries = label_data.index
    exits = label_data['PositionEnd']
    entry_df = refined_data.df['Close'].loc[entries].reset_index().rename(columns = {'Close':'EntryPrice','DateTime':'EntryDate'})
    exit_df = refined_data.df['Close'].loc[exits].reset_index().rename(columns = {'Close':'ExitPrice','DateTime':'ExitDate'})
    side_df = label_data[['Side']].reset_index().drop(columns=['DateTime'])
    trades_df = pd.concat([entry_df,exit_df,side_df],axis = 1)
    trades_df['Profit'] = trades_df['Side']*(trades_df['ExitPrice']-trades_df['EntryPrice'])
    trades_df['NetProfit'] = trades_df['Profit'].cumsum(skipna = False)
    return trades_df.copy()


