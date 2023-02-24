import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller


def ols_regress_coeff(log_prices):

    """
    Granger test using OLS regression
    :param log_prices: log price data for two assets
    :return: nested dictionary with p-value as index and portfolio weights as value dictionary
    """

    model0 = OLS(log_prices[log_prices.columns[0]], add_constant(log_prices[log_prices.columns[1]])).fit()
    model1 = OLS(log_prices[log_prices.columns[1]], add_constant(log_prices[log_prices.columns[0]])).fit()

    test_stat, p_val = [], []
    for model in [model0, model1]:
        adf_res = adfuller(model.resid)
        test_stat.append(adf_res[0])
        p_val.append(adf_res[1])
    adf_res_df = pd.DataFrame({'Test statistic': test_stat,
                               'p-value': p_val}, index=log_prices.columns)
    if adf_res_df.loc[log_prices.columns[0], 'Test statistic'] < adf_res_df.loc[log_prices.columns[1], 'Test statistic']:
        return {adf_res_df.loc[log_prices.columns[0], 'p-value']:
                    {f'{log_prices.columns[0]} Weight': 1, f'{log_prices.columns[1]} Weight': -1*model0.params[0]}}
    else:
        return {adf_res_df.loc[log_prices.columns[1], 'p-value']:
                    {f'{log_prices.columns[0]} Weight': -1*model0.params[0], f'{log_prices.columns[1]} Weight': 1}}

def select_p(log_returns):

    """
    Calculates order for VAR model
    :param log_returns: log returns data for two assets
    :return: minimum order for any of four criterion, AIC, BIC, FPE, HQIC
    """

    aic, bic, fpe, hqic = [], [], [], []
    model = VAR(train_df)
    p = np.arange(1,60)
    for i in p:
        result = model.fit(i)
        aic.append(result.aic)
        bic.append(result.bic)
        fpe.append(result.fpe)
        hqic.append(result.hqic)
    lags_metrics_df = pd.DataFrame({'AIC': aic,
                                    'BIC': bic,
                                    'HQIC': hqic,
                                    'FPE': fpe},
                                   index=p)
    return min(lags_metrics_df.idxmin(axis=0))

def granger_causation_matrix(log_prices, test = 'ssr_chi2test'):

    """
    Check Granger Causality of all possible combinations of the time series.
    :param log_prices: log prices data for two assets
    :return: dictionary with Pair name as key and minimum p value
    """

    log_returns = log_prices.shift.diff()
    p = select_p(log_returns.dropna())
    model = VAR(log_returns.dropna())
    model_fitted = model.fit(p)
    out = durbin_watson(model_fitted.resid)
    df = pd.DataFrame(np.zeros((len(log_prices.columns), len(log_prices.columns))),
                      columns=log_prices.columns, index=log_prices.columns)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(log_prices[[r, c]], p, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1],5) for i in range(p)]
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in data.columns]
    df.index = [var + '_y' for var in data.columns]
    if min(df.min()) < alpha:
        return {f'{df[df[(df.min()).idxmin()]==min(df.min())].index}-{(df.min()).idxmin()}':min(df.min())}, {col : round(val, 3) for col, val in zip(log_prices.columns, out)}

def johansen_test(log_prices, det_order=0, k_ar_diff=1, alpha=0.01):

    assert alpha in [0.05, 0.01], "Only use significance of 1pct or 5pct"

    """
    Johansen test for any number of assets preferably only three
    :param log_prices: log price data
    :param det_order: deterministic terms, default constant
    :param k_ar_diff: number of lagged differences in the model, default 1
    :param alpha: significance level required
    :return: dictionary with ticker names as key and stationary portfolio weights as values,
             returns nothing if anyone of null hypothesis cannot be rejected
    """

    j_obj = coint_johansen(log_prices, det_order=det_order, k_ar_diff=k_ar_diff)
    data = pd.DataFrame(data=j_obj.cvm, index=range(len(log_prices.columns)-1, -1, -1),
                        columns=["10pct", "5pct", "1pct"])
    data["Test"] = j_obj.lr1

    if alpha==0.01 and sum(data["Test"] < data["1pct"]) == 0:
        weight_list = j_obj.evec[j_obj.eig.argmax()]
        return {name: weight for name, weight in zip(log_prices.columns, weight_list)}

    if alpha==0.05 and sum(data["Test"] < data["5pct"]) == 0:
        weight_list = j_obj.evec[j_obj.eig.argmax()]
        return {name: weight for name, weight in zip(log_prices.columns, weight_list)}