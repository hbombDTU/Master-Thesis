import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def scale_data(df: pd.DataFrame,
               columns: list):
    df_scaled = df.copy()

    scaler = MinMaxScaler()
    df_scaled[columns] = scaler.fit_transform(df[columns])

    # Number of observations
    N = df_scaled.shape[0]

    # reference maximum likelihood regression with beta distributed dependent variables
    df_scaled[columns] = df_scaled[columns].apply(lambda x: (x * (N - 1) + 0.5) / N)

    return df_scaled, scaler


def preprocessing(df: pd.DataFrame,
                 air_feature='Air temp Average',
                 gti_feature='GTI Average [W]'):
    # Pre-processing

    # Columns for scaling
    columns_tran = df.columns[:-1]

    # Remove outliers
    outliers_idx = np.where(df[air_feature] < 17)[0] 
    outliers_time = df.iloc[outliers_idx, :].index 
    df_pre = df.drop(outliers_time)  

    # Drop dark hours
    nodark = (df_pre[gti_feature] > 0)
    df_prenodark = df_pre[nodark]

    return df_prenodark


def input_matrix(df: pd.DataFrame,
                 feature_names: list):

    X_df = pd.DataFrame(data=df[feature_names].values, columns=[feature_names], index=df.index)

    # add intercept
    X_df['intcpt'] = np.ones(X_df.shape[0])

    return X_df


def init_params(y_true: np.ndarray,
                X: pd.DataFrame,
                uncen_idx: np.ndarray,
                ):

    y = y_true[uncen_idx]

    init_reg = LinearRegression(fit_intercept=False).fit(X[uncen_idx],y)
    b0 = init_reg.coef_

    # initial predictions
    y_pred = init_reg.predict(X[uncen_idx])

    # compute initial variance
    resid = y - y_pred
    resid_var = np.var(resid)
    s0 = np.std(resid)

    params0 = np.append(b0, s0)

    return params0


def censored_split(df: pd.DataFrame,
                   X: pd.DataFrame,
                   cen_idx: list,
                   uncen_idx: list,
                   power_feature='Power Actual [kW]'):

    # split features into censored and uncensored index
    x_mid = X[uncen_idx] # uncensored
    x_right = X[cen_idx] # right-censored

    y_mid = df[power_feature][uncen_idx]
    y_right = df[power_feature][cen_idx]

    xs = [x_mid, x_right]
    ys = [y_mid, y_right]

    return xs, ys

  
def print_results(y_true: np.ndarray,
                  y_latent: np.ndarray,
                  uncen_idx: list):

    # compute performance metrics
    print('The Coefficient of determination (R-squared) = {:.4f}'.format(
        r2_score(y_true[uncen_idx], y_latent[uncen_idx])))

    print('The mean absolute error (MAE)                = {:.4f}'.format(
        mean_absolute_error(y_true[uncen_idx], y_latent[uncen_idx])))

    print('The root mean squared error (RMSE)           = {:.4f}'.format(
        mean_squared_error(y_true[uncen_idx], y_latent[uncen_idx], squared=False)))

    return None

  
def rev_min_max_func(scaled_val, df, name):
    max_val = max(df[name])
    min_val = min(df[name])
    N = df.shape[0]
    scaled_val_post = (scaled_val * N - 0.5) / (N - 1)
    og_val = (scaled_val_post * (max_val - min_val)) + min_val

    return og_val
