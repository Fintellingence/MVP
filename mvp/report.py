import pandas as pd
import mvp
import matplotlib.pyplot as plt

def trade_book(primary_model, operation_parameters):
    close_data = primary_model.feature_data['Close']
    label_data = mvp.labels.Labels(primary_model.events, close_data, operation_parameters).label_data
    entries = label_data.index
    exits = label_data['PositionEnd']
    entry_df = close_data.loc[entries].reset_index().rename(columns = {'Close':'EntryPrice','DateTime':'EntryDate'})
    exit_df = close_data.loc[exits].reset_index().rename(columns = {'Close':'ExitPrice','DateTime':'ExitDate'})
    side_df = label_data[['Side']].reset_index().drop(columns=['DateTime'])
    trades_df = pd.concat([entry_df,exit_df,side_df],axis = 1)
    trades_df['Profit'] = trades_df['Side']*(trades_df['ExitPrice']-trades_df['EntryPrice'])
    trades_df['NetProfit'] = trades_df['Profit'].cumsum(skipna = False)
    return trades_df.copy()

def avg_holding_time(book):
    return (book['ExitDate']-book['EntryDate']).mean()

def gross_profit(book):
    return book[book['Profit']>0]['Profit'].sum() 

def gross_loss(book):
    return book[book['Profit']<0]['Profit'].sum() 

def net_profit(book):
    return gross_profit(book) + gross_loss(book)

def best_trade(book):
    return book[book['Profit'] == book['Profit'].max()].drop(columns=['NetProfit'])

def worst_trade(book):
   return book[book['Profit'] == book['Profit'].min()].drop(columns=['NetProfit'])

def time_range(primary_model):
    return primary_model.index[0], primary_model.index[-1]

def plot_value(book):
    book['NetProfit'].plot()
    plt.show()
    return 0

def report(primary_model, operation_parameters):
    book = trade_book(primary_model, operation_parameters)
    operation_frequency = 'minute-1'
    print('++++++++++++++++++++')
    print('Asset: '+primary_model.symbol)
    print('Average Holding Time: '+str(avg_holding_time(book)))
    print('Operation frequency: '+operation_frequency)
    print('Best trade: ')
    print(best_trade(book))
    print('Worst trade:')
    print(worst_trade(book))
    print('Order Book:')
    print(book)
    return 0
