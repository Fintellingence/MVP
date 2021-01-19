import subprocess
import pandas as pd
import sqlite3
from zipfile import ZipFile

def two_digits_range(n):
    return [two_digits_string(element) for element in range(1,n+1)] 

def two_digits_string(n):
    if n<10:
        return '0'+str(n)
    else:
        return str(n)

def generate_data_fundos_yearData(from_year,to_year):
    CNPJ_df = pd.DataFrame()
    for year in range(from_year,to_year + 1):
        CNPJ_year_df = pd.read_csv(ZipFile('Data/cda_fi_'+str(year)+'.zip').open('cda_fi_PL_'+str(year)+'.csv'), delimiter=';',encoding = "ISO-8859-1")[['TP_FUNDO','CNPJ_FUNDO','DENOM_SOCIAL']]
        CNPJ_df = pd.concat([CNPJ_df,CNPJ_year_df])
    CNPJ_df.columns = [['TP_FUNDO','CNPJ_FUNDO','DENOM_SOCIAL']]
    return CNPJ_df.drop_duplicates().reset_index().drop(columns='index')

def generate_data_fundos_yearMonthData(from_year,to_year):
    CNPJ_df = pd.DataFrame()
    for year in range(from_year,to_year + 1):
            for yearMonth in [str(year)+month for month in two_digits_range(12)]:
                CNPJ_yearMonth_df = pd.read_csv(ZipFile('Data/'+yearMonth[:4]+'/cda_fi_'+yearMonth+'.zip').open('cda_fi_PL_'+yearMonth+'.csv'), delimiter=';',encoding = "ISO-8859-1")[['TP_FUNDO','CNPJ_FUNDO','DENOM_SOCIAL']]
                CNPJ_df = pd.concat([CNPJ_df,CNPJ_yearMonth_df])
    CNPJ_df.columns = [['TP_FUNDO','CNPJ_FUNDO','DENOM_SOCIAL']]
    return CNPJ_df.drop_duplicates().reset_index().drop(columns='index')

def create_database_fundos(from_year,to_year,split_year):
    CNPJ_df = pd.DataFrame(columns=[['TP_FUNDO','CNPJ_FUNDO','DENOM_SOCIAL']])
    CNPJ_df = pd.concat([CNPJ_df,generate_data_fundos_yearData(from_year,split_year),generate_data_fundos_yearMonthData(split_year+1,to_year)]).drop_duplicates().reset_index().drop(columns='index')
    conn = sqlite3.connect('Data/Fundos.db')
    CNPJ_df.to_sql('Fundos Brasileiros', con =conn, if_exists='replace')
    conn.close()
    pass

def create_fundo_database_entry(fundo_CNPJ):
    from_year = 2005
    split_year = 2017
    to_year = 2019
    fundo_df = pd.DataFrame()
    for year in range(from_year,split_year + 1)[:1]:
        fundo_year_df = pd.read_csv(ZipFile('Data/cda_fi_'+str(year)+'.zip').open('cda_fi_BLC_4_'+str(year)+'.csv'), delimiter=';',encoding = "ISO-8859-1")[['CNPJ_FUNDO','DT_COMPTC','TP_APLIC',\
                'QT_VENDA_NEGOC','VL_VENDA_NEGOC','QT_AQUIS_NEGOC','VL_AQUIS_NEGOC','QT_POS_FINAL','VL_MERC_POS_FINAL','CD_ATIVO']]
        fundo_df = pd.concat([fundo_df,fundo_year_df])
    for year in range(split_year + 1,to_year + 1)[:1]:
            for yearMonth in [str(year)+month for month in two_digits_range(12)]:
                fundo_yearMonth_df = pd.read_csv(ZipFile('Data/'+yearMonth[:4]+'/cda_fi_'+yearMonth+'.zip').open('cda_fi_BLC_4_'+yearMonth+'.csv'), delimiter=';',encoding = "ISO-8859-1")[['CNPJ_FUNDO','DT_COMPTC','TP_APLIC',\
                'QT_VENDA_NEGOC','VL_VENDA_NEGOC','QT_AQUIS_NEGOC','VL_AQUIS_NEGOC','QT_POS_FINAL','VL_MERC_POS_FINAL','CD_ATIVO']]
                fundo_df = pd.concat([fundo_df,fundo_yearMonth_df])
    fundo_df.columns = [['CNPJ_FUNDO','DT_COMPTC','TP_APLIC','QT_VENDA_NEGOC','VL_VENDA_NEGOC','QT_AQUIS_NEGOC','VL_AQUIS_NEGOC','QT_POS_FINAL','VL_MERC_POS_FINAL','CD_ATIVO']]
    fundo_df = fundo_df.reset_index().drop(columns='index')
    print(fundo_df.loc[fundo_df['CNPJ_FUNDO'].isin(fundo_CNPJ)])

create_fundo_database_entry('01.541.649/0001-82')