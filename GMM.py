import math
import pandas as pd

from sklearn.mixture import GaussianMixture

# function definition to compute magnitude o f the vector
def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))


def compute_PCA(df, features, number):
    pca = PCA(n_components=number)
    X_all = pd.DataFrame(pca.fit_transform(df.loc[:, features].values))

    return X_all


def train_GMM(X_train):
    gmm = GaussianMixture(n_components=2,
                          n_init=1,
                          init_params='kmeans',
                          covariance_type='full').fit(X_train)
    y_ = gmm.predict(X_train)

    # Relabel the predictions to correct labels
    rev = False
    if magnitude(gmm.means_[0]) < magnitude(gmm.means_[1]):
        y_ = np.array([1 if y == 0 else 0 for y in y_])
        rev = True

    if rev:
        gmm_df = pd.DataFrame(gmm.predict_proba(X_train), columns=['ucen', 'cen'])
    else:
        gmm_df = pd.DataFrame(gmm.predict_proba(X_train), columns=['cen', 'uncen'])

    gmm_df['predict'] = y_

    unsure_idx = (gmm_df.predict == 0) & (gmm_df['cen'] < 0.7)
    unsure_series = pd.Series(unsure_idx)
    unsure_values = unsure_series[unsure_idx].index.values

    gmm_df.iloc[unsure_values, 2] = 1
    y_ = gmm_df.iloc[:, 2]
    return y_


def baseline(error_square):
    return [0 if x_value > 0.025 else 1 for x_value in error_square.values]

