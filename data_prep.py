import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

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


def preprocessing(df: pd.DataFrame):
    # Pre-processing

    # Columns for scaling
    columns_tran = df.columns[:-1]

    # Remove outliers
    outliers_idx = np.where(df['Air temp Average'] < 17)[0]  # np.where(df_scaled['Mod_Temp Average'] <= 0.2)[0]
    outliers_time = df.iloc[outliers_idx, :].index  # df_scaled.iloc[outliers_idx, :].index
    df_pre = df.drop(outliers_time)  # df_scaled.drop(outliers_time)

    # Drop dark hours
    nodark = (df_pre['GTI Average [W]'] > 0)  # df_scaled['Power Actual [kW]'] > min(df_scaled['Power Actual [kW]'])
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
                   uncen_idx: list):

    # split features into censored and uncensored index
    x_mid = X[uncen_idx] # uncensored
    x_right = X[cen_idx] # right-censored

    y_mid = df['Power Actual [kW]'][uncen_idx]
    y_right = df['Power Actual [kW]'][cen_idx]

    xs = [x_mid, x_right]
    ys = [y_mid, y_right]

    return xs, ys