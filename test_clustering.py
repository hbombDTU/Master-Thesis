import numpy as np
import pandas as pd

from data_prep import *

# Import data
df = pd.read_excel('SCADA data for forecasting.xlsx',engine='openpyxl', index_col='t_stamp',sheet_name='11_21')
df.rename(columns = {'GHI\nAverage\n[W]': 'GHI Average [W]', 'GTI\nAverage\n[W]': 'GTI Average [W]', 'Power Actual\n[kW]': 'Power Actual [kW]'}, inplace=True)
df['VALID FLAG'] = df['VALID FLAG'].astype('category')

df.rename(columns={'Power Theoretical [kW]': 'Enfor'}, inplace=True)
FeatureList_pre = preprocessing(df)

columns_trans = FeatureList_pre.columns[:-1]
FeatureList_pre, scaler = scale_data(FeatureList_pre,columns_trans)