import numpy as np

from scipy.optimize import minimize
from scipy.special import gamma
from scipy.special import beta  # complete beta function (a,b)
from scipy.special import betainc  # incomplete beta function (a,b,x)
from scipy.special import betaln
from scipy.special import betaincinv

from data_prep import init_params, censored_split

def jfst_pdf(u, a, b, sigma=1):
    if not isinstance(u, float):
        u = np.array(u)

    v = a + b
    mu = 0  # np.mean(u)
    z = (u - mu) / sigma

    C_ab = lambda a, b: 2 ** (v - 1) * beta(a, b) * np.sqrt(v)

    f_pdf = C_ab(a, b) ** -1 / sigma \
            * (1 + (z) / np.sqrt(v + (z) ** 2)) ** (a + 0.5) \
            * (1 - (z) / np.sqrt(v + (z) ** 2)) ** (b + 0.5)

    return f_pdf


def jfst_cdf(u, a, b, sigma=1):

    if not isinstance(u, float):
        u = np.array(u)
    v = a + b
    mu = 0
    z = (u - mu) / sigma
    x = (1 + (z) / np.sqrt(v + (z) ** 2)) / 2

    assert any(np.isnan(betainc(a, b, x))) == False, 'There are NaN values'
    I_x = lambda a, b, x: betainc(a, b, x)

    F_cdf = I_x(a, b, x)
    return F_cdf


def jfst_logpdf(u, a, b, sigma=1):
    if not isinstance(u, float):
        u = np.array(u)

    v = a + b
    mu = 0
    z = (u - mu) / sigma

    f_logpdf = - ((v - 1) * np.log(2) + betaln(a, b) + 1 / 2 * np.log(v)) - np.log(sigma) \
               + (a + 0.5) * np.log(1 + (z) / np.sqrt(v + (z) ** 2)) \
               + (b + 0.5) * np.log(1 - (z) / np.sqrt(v + (z) ** 2))

    return f_logpdf


def jfst_logcdf(u, a, b, sigma=1):
    if not isinstance(u, float):
        u = np.array(u)

    v = a + b
    mu = 0
    z = (u - mu) / sigma
    x = (1 + z / np.sqrt(v + (z) ** 2)) / 2

    f_logcdf = np.log(betainc(a, b, x))

    return f_logcdf


def jfst_inv_cdf(u, a, b, sigma, mu=0):
    if not isinstance(u, float):
        u = np.array(u)
    v = a + b

    inv_Ix = lambda a, b, u: betaincinv(a, b, u)


    x_ = inv_Ix(a, b, u)
    x_array = np.sqrt(v) * (2 * x_ - 1) / np.sqrt(1 - (2 * x_ - 1) ** 2)

    x_final = x_array * sigma

    return x_final


def jfst_NLL(xs, ys, params, a_shape, b_shape):
    x_mid, x_right = xs
    y_mid, y_right = ys

    b = params[:-1]  # inlcudes intcpt
    s = params[-1]

    right = np.dot(x_right,b) - y_right
    # Compute the CDF; if SF then right = y_right - npdot(x_right,b)
    log_cdf = jfst_logcdf(right, a=a_shape, b=b_shape,sigma=s)
    cens_sum = log_cdf.sum()

    # Compute the PDF of uncensored
    mid =y_mid - np.dot(x_mid, b)
    log_pdf = jfst_logpdf(mid, a=a_shape, b=b_shape, sigma=s)  ### may need to reformulate
    mid_sum = log_pdf.sum()

    loglik = cens_sum + mid_sum - np.log(s)

    return - loglik


def expected_value(a, b, s):
    e_t = (a - b) * np.sqrt(a + b) / 2 * gamma(a - 1 / 2) * gamma(b - 1 / 2) / (gamma(a) * gamma(b)) * s

    return e_t


def compute_pred(optm_param, X):
    coef = optm_param.x[:-1]
    s = optm_param.x[-1]
    y = np.dot(X, coef)

    return y


def jfst_model_train(y_train,
                     X_train,
                     X_test,
                     cen_idx_train,
                     uncen_idx_train):

    # find optimal a, b shape parameters
    a_bnd = [10, 15]
    b_bnd = [10, 15]
    sigma = 0.02
    N = 10

    a_values, b_values, s_values, NLL_min, NLL_values, best_a, best_b, best_s = find_opt(a_bnd=a_bnd,
                                                                                         b_bnd=b_bnd,
                                                                                         sigma=sigma,
                                                                                         N=N,
                                                                                         cen_idx=cen_idx_train,
                                                                                         uncen_idx=uncen_idx_train,
                                                                                         df=df_train,
                                                                                         X_df=X_train,
                                                                                         # method='Powell'
                                                                                         print_result=False
                                                                                         )

    a_shape = best_a  # 10.0#best_a # 10.0 #1.4
    b_shape = best_b  # 13.0 #best_b # 13.0 #3.43
    s_shape = best_s  # 0.08766455381115859#best_s # 0.08766455381115859#0.015

    # Initialize parameters: beta and variance
    params0 = init_params(y_train,
                          X_train,
                          uncen_idx_train,
                          )

    # censored and uncensored split arrays
    xs, ys = censored_split(df_train, X_train, cen_idx_train, uncen_idx_train)

    bnds = [(None, None) for i, _ in
            enumerate(X_train.columns)]  # ((None, None), (None, None), (None, None), (0, None))
    bnds.append((0.0001, None))  # bounds for sigma

    result = minimize(lambda params: jfst_NLL(xs, ys, params, a_shape=a_shape, b_shape=b_shape), params0,
                      method='L-BFGS-B',
                      bounds=bnds,
                      )

    # Compute predictions
    y_latent = compute_pred(result, X_test)
    y_train = compute_pred(result, X_train)

    neg_idx = y_latent < 0
    median = jfst_inv_cdf(0.5, best_a, best_b, best_s)
    ev = expected_value(best_a, best_b, best_s)
    y_latent[neg_idx] = min(y_test)
    y_latent_median = y_latent + median
    y_latent_exp = y_latent + ev

    best_opt = [best_a, best_b, best_s]
    return [y_latent, y_latent_median, y_latent_exp], y_train, best_opt

