import numpy as np
import pandas as pd


def ticker_n_smallest_distance(norm_prices, n=3):
    """
    Get the names of first n pairs with smallest distance
    :param data: log price data
    :param n: no. of pairs
    :return: dictionary with ticker pairs as value
    """
    
    distance_df = pd.DataFrame(data=calculate_distances(norm_prices).values(), index=calculate_distances(norm_prices).keys(), columns=["Euclidean Distance"])
    ticker_list = distance_df["Euclidean Distance"].index[0:n].to_list()

    return {count+1:parse_pair(pair) for count, pair in enumerate(ticker_list)}

def normalise_prices(log_prices):
    """
    Normalise log prices
    :param log_prices: log price data
    :return: dataframe with normalised prices
    """

    norm_prices = pd.DataFrame(index=log_prices.index)

    for column in log_prices.columns:
        minimum = np.min(log_prices[column])
        maximum = np.max(log_prices[column])
        norm_prices[column] = (log_prices[column] - minimum)/(maximum-minimum)
    return norm_prices

def calculate_spread(norm_prices):
    assert len(norm_prices.columns) == 2, "Normalised Prices of pair of assets"
    """
    Calculate spread for a pair of assets
    :param norm_prices: 
    :return: 
    """
    
    spread_df = pd.DataFrame(index=norm_prices.index)
    s1 = norm_prices.columns[0]
    s2 = norm_prices.columns[1]
    spread_df[f'{s1}-{s2}'] = norm_prices[s1] - norm_prices[s2]

    return spread_df

def calculate_zero_crossings(norm_prices):
    """
    Calculaed number of zero crossings in each pair
    :param norm_prices: normalised log prices of pairs
    :return: sorted dicionary (in ascending order) with ticker name as keys and number of zero crossings as value
    """

    from collections import Counter
    
    zero_dict = {}

    for column in norm_prices.columns:
        my_list = norm_prices[column].to_list()
        zero_dict[column] = (np.diff(np.sign(my_list)) != 0).sum() - Counter(my_list)[0]

    # sort dictionary
    sorted_zero_dict = {k: v for k, v in sorted(zero_dict.items(), key=lambda item: item[1])}

    return sorted_zero_dict

def calculate_spread_variance(norm_prices):
    """
    Calculates the spread variance
    :param norm_prices: normalised price data
    :return: sorted dictionary (in ascending order) with pairs as keys and spread variance as values
    """

    spread_variance = {}

    for s1 in norm_prices.columns:
        for s2 in norm_prices.columns:
            if s1 != s2 and (f'{s1}-{s2}' not in spread_variance.keys()) and (f'{s2}-{s1}' not in spread_variance.keys()):
                spread_variance[f'{s1}-{s2}'] = np.var(norm_prices[s1]-norm_prices[s2])

    # sort dictionary
    sorted_spread_variance = {k: v for k, v in sorted(spread_variance.items(), key=lambda item: item[1])}

    return sorted_spread_variance

def calculate_distances(norm_prices):
    """
    calculate Euclidean distance for each pair of stocks in the dataframe
    :param log_prices: log prices data
    :return: sorted dictionary (in ascending order)
    """

    distances = {}  # dictionary with distance for each pair

    # calculate distances
    for s1 in norm_prices.columns:
        for s2 in norm_prices.columns:
            if s1 != s2 and (f'{s1}-{s2}' not in distances.keys()) and (f'{s2}-{s1}' not in distances.keys()):
                dist = np.sqrt(np.sum((norm_prices[s1] - norm_prices[s2]) ** 2))  # Euclidean distance
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

def calculate_halflife(spread):
    """
    calculate half-life of mean reversion of the spread
    :param log_prices: log_prices of pair of assets
    :return: halflife of the spread
    """

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    ylag = spread.shift()
    deltay = spread - ylag
    ylag.dropna(inplace=True)
    deltay.dropna(inplace=True)

    res = OLS(deltay, add_constant(ylag)).fit()
    halflife = -np.log(2) / res.params[0]

    return halflife


def calculate_metrics(sorted_distances, train_data, N=5):
    """
    calculate metrics for N pairs with the smallest Euclidean distance
    return dataframe of results
    """
    from statsmodels.tsa.stattools import adfuller

    pairs = [k for k, v in sorted_distances.items()][:N]

    cols = ['Euclidean distance', 'ADF p-value', 'Spread SD', 'Half-life of mean reversion', '% days within 2-SD band']
    results = pd.DataFrame(index=pairs, columns=cols)

    for pair in pairs:
        s1, s2 = parse_pair(pair)
        spread = train_data[s1] - train_data[s2]
        results.loc[pair]['Euclidean distance'] = np.sqrt(np.sum((spread) ** 2))
        results.loc[pair]['ADF p-value'] = adfuller(spread)[1]
        results.loc[pair]['Spread SD'] = spread.std()
        results.loc[pair]['Half-life of mean reversion'] = calculate_halflife(spread)
        results.loc[pair]['% days within 2-SD band'] = (abs(spread) < 2 * spread.std()).sum() / len(spread) * 100

    return results

def plot_pairs(train_data, test_data, N=5):
    """
    plots the cumulative return on the spread for each pair
    :param data_train: training price data
    :param cumret_train: training cumulative returns data
    :param cumret_test: testing cumulative returns data
    :param N: no. of pairs
    """
    import matplotlib.pyplot as plt
    
    # calculate Euclidean distances for each pair
    sorted_distances = calculate_distances(train_data)
    pairs = [k for k, v in sorted_distances.items()][:N]

    for pair in pairs:
        s1, s2 = parse_pair(pair)
        spread_train = train_data[s1] - train_data[s2]
        spread_test = test_data[s1] - test_data[s2]
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