import numpy as np
import pandas as pd
from itertools import combinations
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller



def partner_selection_metrics(prices_train, cumret_train):
    """
    Calculates partner selection metrics
    :param prices_train: training price data
    :param cumret_train: training cumulative returns data
    :return: dataframe with Pairs and metrics
    """
    pair_metrics = pd.DataFrame(columns=['MDM', 'ADF', 'G_12', 'G_21', 'MFR'])
    for s1 in cumret_train.columns:
        for s2 in cumret_train.columns:
            if (s1!=s2) and (f'{s1}-{s2}' not in pair_metrics.index) and (f'{s2}-{s1}' not in pair_metrics.index):
                # distance
                pair_metrics.loc[f'{s1}-{s2}', ['MDM']] = ((cumret_train[s1] - cumret_train[s2]) ** 2).sum()
                # ADF test
                spread = prices_train[s1] / prices_train[s2]
                pair_metrics.loc[f'{s1}-{s2}', ['ADF']] = adfuller(spread)[1]

    return pair_metrics

def spearman_rank_correlation(log_returns, n=3):
    """
    Calculate sum of all pairwise Spearman correlation for all possible quadruples.
    :param log_returns: log returns data
    :param n: no. of quadruples
    :return: returns dataframe with top n pair quadruples with highest sum
    """
    ranked_returns = log_returns.rank()
    quadruples = pd.DataFrame()
    for s1 in ranked_returns.columns:
        for comb in list(combinations(ranked_returns.columns, 4)):
            if s1 not in list(comb):
                rho = 0
                rho += ranked_returns[[s1, comb[0]]].corr(method="spearman")[s1][1]
                rho += ranked_returns[[s1, comb[1]]].corr(method="spearman")[s1][1]
                rho += ranked_returns[[s1, comb[2]]].corr(method="spearman")[s1][1]
                rho += ranked_returns[[s1, comb[3]]].corr(method="spearman")[s1][1]
                quadruples.loc[f'{s1} with {comb}', 'Spearman Rank Sum'] = rho

    return quadruples.sort_values(by='Spearman Rank Sum', ascending=False).head(n)

def kendall_rank_correlation(log_returns, n):
    """
    Calculate sum of all pairwise Kendall correlation for all possible quadruples.
    :param ranked_returns: returns data
    :param n: no. of pairs
    :return: returns dataframe with top n 
    """

    ranked_returns = log_returns.rank()
    quadruples = pd.DataFrame()
    for s1 in ranked_returns.columns:
        for comb in list(combinations(ranked_returns.columns, 4)):
            if s1 not in list(comb):
                tau = 0
                tau += stats.kendalltau(ranked_returns[s1], ranked_returns[comb[0]])[0]
                tau += stats.kendalltau(ranked_returns[s1], ranked_returns[comb[1]])[0]
                tau += stats.kendalltau(ranked_returns[s1], ranked_returns[comb[2]])[0]
                tau += stats.kendalltau(ranked_returns[s1], ranked_returns[comb[3]])[0]
                quadruples.loc[f'{s1} with {comb}', 'Kendall Rank Sum'] = tau

    return quadruples.sort_values(by='Kendall Rank Sum', ascending=False).head(n)