import numpy as np
import pandas as pd

from data_prep import *
from GMM import *

# Import data
df_pre = pd.read_excel('SCADA data for forecasting.xlsx',engine='openpyxl', index_col='t_stamp',sheet_name='11_21')
df_pre.rename(columns = {'GHI\nAverage\n[W]': 'GHI Average [W]', 'GTI\nAverage\n[W]': 'GTI Average [W]', 'Power Actual\n[kW]': 'Power Actual [kW]'}, inplace=True)
df_pre['VALID FLAG'] = df_pre['VALID FLAG'].astype('category')

df_pre.rename(columns={'Power Theoretical [kW]': 'Enfor'}, inplace=True)

# Preprocess and scale data
df = preprocessing(df_pre)

columns_trans = df.columns[:-1]
df, scaler = scale_data(df,columns_trans)

y_true = df['VALID FLAG']
# Compute squared error
features = ['Enfor', 'Power Actual [kW]']
df['error_square'] = compute_error(df, features)

# Compute baseline predictions
y_baseline = baseline(df['error_square'])
print_results_gmm(y_true,y_baseline)

# Prepare the training dataset
features_gmm = ['error_square']
X_gmm = compute_PCA(df,
                    features_gmm,
                    number=1)

# Compute GMM predictions
y_gmm = train_GMM(X_gmm)

print_results_gmm(y_true,y_gmm)

print('Done')
