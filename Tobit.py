from scipy.optimize import minimize
from scipy.stats import norm
import math

def tobit_NLL(xs, ys, params):
    x_mid, x_right = xs
    y_mid, y_right = ys

    b = params[:-1]  # inlcudes intcpt
    s = params[-1]

    right = np.dot(x_right,b)-y_right#(np.dot(x_right, b) - y_right)
    right_stats = right / s
    # Compute the CDF; if SF then right = y_right - npdot(x_right,b)
    log_norm_cdf = np.log(norm.cdf(right_stats))#norm.logsf(right,loc=0,scale=s) #norm.logcdf(right_stats) # weirdly norm.logcdf doesn't perform so well
    cens_sum = log_norm_cdf.sum()

    # Compute the PDF of uncensored
    mid = (y_mid - np.dot(x_mid, b))
    mid_stats = mid / s
    log_norm_pdf = norm.logpdf(mid_stats) - math.log(max(np.finfo('float').resolution, s))
    mid_sum = log_norm_pdf.sum()

    loglik = cens_sum + mid_sum

    return - loglik


def tobit_NLL_der(xs, ys, params):
    x_mid, x_right = xs
    y_mid, y_right = ys

    b = params[:-1]
    s = params[-1]  # this is variance

    beta_jac = np.zeros(
        len(b))  # this is adding the partial derivative of \beta, it is the length of b because it is the length of the coefficients
    sigma_jac = 0  # this is the second equation

    right = (np.dot(x_right, b) - y_right)
    right_stats = right / s

    right_pdf = norm.logpdf(right_stats)
    right_cdf = norm.logcdf(right_stats)
    right_frac = np.exp(right_pdf - right_cdf)  # to get f/(1-F) where we assume f and F is not a log function

    beta_right = np.dot(right_frac, x_right / s)
    beta_jac += beta_right

    right_sigma = np.dot(right_frac, right_stats)
    sigma_jac -= right_sigma

    ### uncensored computation
    mid = y_mid - np.dot(x_mid, b)
    mid_stats = mid / s

    beta_mid = np.dot(mid_stats, x_mid / s)
    beta_jac += beta_mid

    mid_sigma = (np.square(mid_stats) - 1).sum()  # -1 is from (T-S)/(2\sigma^2)
    sigma_jac += mid_sigma / s

    combo_jac = np.append(beta_jac, sigma_jac / s)
    return - combo_jac  # because we are doing NLL


def tobit_model_train(y_train,
                      X_train,
                      X_test,
                      cen_idx_train,
                      uncen_idx_train):
    # Initialize parameters: beta and variance
    params0 = init_params(y_train,
                          X_train,
                          uncen_idx_train,
                          )

    # censored and uncensored split arrays
    xs, ys = censored_split(df_train, X_train, cen_idx_train, uncen_idx_train)

    bnds = [(None, None) for i, _ in
            enumerate(X_train.columns)]  # ((None, None), (None, None), (None, None), (0, None))
    bnds.append((0.00001, None))  # bounds for sigma

    result = minimize(lambda params: tobit_NLL(xs, ys, params), params0, method='L-BFGS-B',
                      jac=lambda params: tobit_NLL_der(xs, ys, params),
                      bounds=bnds
                      )
    # Compute predictions
    y_latent = compute_pred(result, X_test)
    y_train = compute_pred(result, X_train)

    neg_idx = y_latent < 0
    y_latent[neg_idx] = min(y_test)

    return y_latent, y_train, result.x[-1]