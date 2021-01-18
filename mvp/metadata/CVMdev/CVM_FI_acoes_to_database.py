import subprocess
import pandas as pd
from zipfile import ZipFile


def generate_CNPJ_yearData(from_year,to_year):
    CNPJ_df = pd.DataFrame()
    for year in range(from_year,to_year + 1):
        CNPJ_year_df = pd.read_csv(ZipFile('Data/cda_fi_'+str(year)+'.zip').open('cda_fi_PL_'+str(year)+'.csv'), delimiter=';',encoding = "ISO-8859-1")['CNPJ_FUNDO']
        CNPJ_df = pd.concat([CNPJ_df,CNPJ_year_df])
    CNPJ_df.columns = ['CNPJ_FUNDO']
    return CNPJ_df.drop_duplicates().reset_index().drop(columns='index')

def generate_CNPJ_yearMonthData(from_yearMonth,to_yearMonth):
    CNPJ_df = pd.DataFrame()
    for year in range(from_year,to_year + 1):
        CNPJ_year_df = pd.read_csv(ZipFile('Data/cda_fi_'+str(year)+'.zip').open('cda_fi_PL_'+str(year)+'.csv'), delimiter=';',encoding = "ISO-8859-1")['CNPJ_FUNDO']
        CNPJ_df = pd.concat([CNPJ_df,CNPJ_year_df])
    CNPJ_df.columns = ['CNPJ_FUNDO']
    return CNPJ_df.drop_duplicates().reset_index().drop(columns='index')

print(generate_CNPJ_yearData(2005,2017))