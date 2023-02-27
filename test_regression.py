import numpy as np
import pandas as pd


from JFST import *
from data_prep import *
from sklearn.model_selection import train_test_split

# Read the data

# Import data
df = pd.read_excel('SCADA data for forecasting.xlsx',engine='openpyxl', index_col='t_stamp', sheet_name='11_21')
df.rename(columns = {'GHI\nAverage\n[W]': 'GHI Average [W]', 'GTI\nAverage\n[W]': 'GTI Average [W]', 'Power Actual\n[kW]': 'Power Actual [kW]'}, inplace=True)
df['VALID FLAG'] = df['VALID FLAG'].astype('category')

# Process the dataset first
df_prenodark = preprocessing(df)

# Columns for scaling
columns_tran = df_prenodark.columns[:-1]

# Define feature names
feature_names = ['GTI Average [W]', 'Mod_Temp Average']

df_train_pre, df_test_pre = train_test_split(df_prenodark, test_size=0.3, random_state=0, shuffle=False)

N = df_test_pre.shape[0]

train_num_cen = df_train_pre[df_train_pre['VALID FLAG']==1].shape[0]
size_uncen = train_num_cen
size_cen = int(size_uncen/0.55*0.45)

new_uncen_idx = df_train_pre[df_train_pre['VALID FLAG']==1][-size_uncen:]
new_cen_idx = df_train_pre[df_train_pre['VALID FLAG']==0][-size_cen:]
new_idx = pd.concat([new_uncen_idx,new_cen_idx])

df_train, scaler = scale_data(df_train_pre.loc[new_idx.index],columns_tran)

df_test = df_test_pre.copy()
df_test[columns_tran] = scaler.transform(df_test_pre[columns_tran])
df_test[columns_tran] = df_test[columns_tran].apply(lambda x: (x*(N-1)+0.5)/N)

X_train = input_matrix(df_train, feature_names)

X_test= input_matrix(df_test, feature_names)
y_train = df_train['Power Actual [kW]']
y_test = df_test['Power Actual [kW]']

cen_idx_train = df_train['VALID FLAG'] == 0
cen_idx_test = df_test['VALID FLAG'] == 0
uncen_idx_train = df_train['VALID FLAG'] == 1
uncen_idx_test = df_test['VALID FLAG'] == 1



# Train and test JFST Model
[y_latent, y_latent_median, y_latent_exp], y_train, best_opt = jfst_model_train(df_train,
                                                                                y_train,
                                                                                X_train,
                                                                                X_test,
                                                                                cen_idx_train,
                                                                                uncen_idx_train)


# print results
print_results(y_true = y_test,
              y_latent= y_latent_median,
              uncen_idx=uncen_idx_test)

print('-'*50)
# Inverse scaled values
y_pred = rev_min_max_func(y_latent_median, df_test_pre, 'Power Actual [kW]')
y_true = df_test_pre['Power Actual [kW]'].values
y_th = df_test_pre['Power Theoretical [kW]'].values

print_results(y_true = y_true,
              y_latent= y_pred,
              uncen_idx=uncen_idx_test)