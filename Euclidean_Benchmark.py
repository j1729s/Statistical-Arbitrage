import numpy as np

def ticker_n_smallest_distance(log_prices, n=3):
    """
    Get the names of first n pairs with smallest distance
    :param data: log price data
    :param n: no. of pairs
    :return: dictionary with ticker pairs as value
    """
    wh
    distance_df = pd.DataFrame(data=calculate_distances(data).values(), index=calculate_distances(data).keys(), columns=["Euclidean Distance"])
    ticker_list = distance_df["Euclidean Distance"].index[0:n].to_list()
    return {count+1:parse_pair(pair) for count, pair in enumerate(ticker_list)}

def demean_and_cal_spread(log_prices):

    """
    Demean and calculate spread between two assets log prices
    :param log_prices: log price data
    :return: dataframe with spread
    """

    spread_df = pd.DataFrame(index=log_prices.index)
    s1 = log_prices.columns[0]
    s2 = log_prices.columns[1]
    log_prices.loc[, s1] = log_prices[s1] - log_prices[s1].mean()
    log_prices.loc[, s2] = log_prices[s2] - log_prices[s2].mean()
    spread_df[f'{s1}-{s2}'] = log_prices[s1] - log_prices[s2]
    return spread_df

def calculate_distances(log_prices):

    """
    calculate Euclidean distance for each pair of stocks in the dataframe
    :param log_prices: log prices data
    :return: sorted dictionary (in ascending order)
    """

    distances = {}  # dictionary with distance for each pair

    # calculate distances
    for s1 in log_prices.columns:
        for s2 in log_prices.columns:
            if s1 != s2 and (f'{s1}-{s2}' not in distances.keys()) and (f'{s2}-{s1}' not in distances.keys()):
                dist = np.sqrt(np.sum((log_prices[s1] - log_prices[s2]) ** 2))  # Euclidean distance
                distances[f'{s1}-{s2}'] = dist

    # sort dictionary
    sorted_distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

    return sorted_distances


def parse_pair(pair):

    """
    :param pair: parse pair string 'S1-S2'
    :return: tickers S1, S2
    """

    dp = pair.find('-')
    s1 = pair[:dp]
    s2 = pair[dp + 1:]

    return s1, s2

def calculate_halflife(log_prices):

    """
    calculate half-life of mean reversion of the spread
    :param log_prices: log_prices of pair of assets
    :return: halflife of the spread
    """

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    spread = log_prices[log_prices.columns[0]] - log_prices[log_prices.columns[1]]
    ylag = spread.shift()
    deltay = spread - ylag
    ylag.dropna(inplace=True)
    deltay.dropna(inplace=True)

    res = OLS(deltay, add_constant(ylag)).fit()
    halflife = -np.log(2) / res.params[0]

    return halflife


def calculate_metrics(sorted_distances, cumret, N=5):
    '''
    calculate metrics for N pairs with the smallest Euclidean distance
    return dataframe of results
    '''
    from hurst import compute_Hc
    from statsmodels.tsa.stattools import adfuller

    pairs = [k for k, v in sorted_distances.items()][:N]

    cols = ['Euclidean distance', 'CADF p-value', 'ADF p-value', 'Spread SD',
            'Num zero-crossings', 'Hurst Exponent', 'Half-life of mean reversion', '% days within 2-SD band']
    results = pd.DataFrame(index=pairs, columns=cols)

    for pair in pairs:
        s1, s2 = parse_pair(pair)
        spread = cumret[s1] - cumret[s2]
        results.loc[pair]['Euclidean distance'] = np.sqrt(np.sum((spread) ** 2))
        results.loc[pair]['CADF p-value'] = cadf_pvalue(s1, s2, cumret)
        results.loc[pair]['ADF p-value'] = adfuller(spread)[1]
        results.loc[pair]['Spread SD'] = spread.std()
        results.loc[pair]['Num zero-crossings'] = ((spread[1:].values * spread[:-1].values) < 0).sum()
        results.loc[pair]['Hurst Exponent'] = compute_Hc(spread)[0]
        results.loc[pair]['Half-life of mean reversion'] = calculate_halflife(spread)
        results.loc[pair]['% days within 2-SD band'] = (abs(spread) < 2 * spread.std()).sum() / len(spread) * 100

    return results

def plot_pairs(data_train, cumret_train, cumret_test, N=5):

    """
    plots the cumulative return on the spread for each pair
    :param data_train: training price data
    :param cumret_train: training cumulative returns data
    :param cumret_test: testing cumulative returns data
    :param N: no. of pairs
    """

    # calculate Euclidean distances for each pair

    sorted_distances12_6 = calculate_distances(data_train)
    pairs = [k for k, v in sorted_distances.items()][:N]

    for pair in pairs:
        s1, s2 = parse_pair(pair)
        spread_train = cumret_train[s1] - cumret_train[s2]
        spread_test = cumret_test[s1] - cumret_test[s2]
        spread_mean = spread_train.mean()  # historical mean
        spread_std = spread_train.std()  # historical standard deviation

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))
        fig.suptitle(f'Spread of {pair} pair', fontsize=16)
        ax1.plot(spread_train, label='spread')
        ax1.set_title('Formation period')
        ax1.axhline(y=spread_mean, color='g', linestyle='dotted', label='mean')
        ax1.axhline(y=spread_mean + 2 * spread_std, color='r', linestyle='dotted', label='2-SD band')
        ax1.axhline(y=spread_mean - 2 * spread_std, color='r', linestyle='dotted')
        ax1.legend()
        ax2.plot(spread_test, label='spread')
        ax2.set_title('Trading period')
        ax2.axhline(y=spread_mean, color='g', linestyle='dotted', label='mean')
        ax2.axhline(y=spread_mean + 2 * spread_std, color='r', linestyle='dotted', label='2-SD band')
        ax2.axhline(y=spread_mean - 2 * spread_std, color='r', linestyle='dotted')
        ax2.legend()