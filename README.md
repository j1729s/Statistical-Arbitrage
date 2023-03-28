# Statistical-Arbitrage

This is my Pairs Trading Library. This implementation is supposed to be a free alternative to the Hudson & Thames Library for Statitical Arbitrage. I hope that this will lead to a profitable Pairs Trading Startegy. But for now this represents a proof of concept using price data of a few tickers. 

### Euclidean Benchmark
We start with the simplest of all pairs trading strategies, which is calculating the Euclidean Distance between normalised price data of assets to form pairs based on the smallest distances. I will later combine this with a Minimum Profit Optimiation strategy that will tll us when to open our position and when to close. 

**STEP 1: Normalise Price Data**
$$P_{\text{norm}}=\frac{P-\text{min}(P)}{\text{max}(P)-\text{min}(P)}$$
This is done in order to bring the price data of each asset to the same scale. 

**STEP 2: Calculate Sum of Squared Distances**
$$\text{SSD}=\Sigma_{t=1}^{N}{(P_{t}^{1}-P_{t}^{2})^{2}}$$
Using Euclidean squared distance on the normalized price time series, $n$ closest pairs of assets are picked.

**STEP 3: Create Strategy for Entry and Exit**

If the difference between the price of elements in a pair diverged by more than a threshold, the positions are opened

a. Long for the element with a lower price in a portfolio

b. Short for the element with a higher price in a portfolio

Most commonly used threshhold is Bollinger Bands i.e., usually 2 standard deviations.

#### Alternative Metrics for Pair Selection
**1. Pairs within the same industry group**

By calculating the Euclidean square distance for each of the pair within the same group, the $n$ closest pairs are selected.

**2. Pairs with a higher number of zero-crossings**

The top $n$ pairs that had the highest number of zero crossings during the formation period are admitted to the portfolio we select.

**3. Pairs with a higher historical standard deviation**

Select top $n$ pairs with the highest variance of the spread.

All of the above approches have been implemented as functions in ```Euclidean_Benchmark.py``` (Also see: ```Euclidean_Test.ipynb```). Working with price data for stocks in the same industry, I would like to combine, pairs that have smallest distances, higher number of zero crossings and higher spread variance, with a Minimum Profit Optimisation Entry-Exit Strategy. This would be my Benchmark Strategy.

### Cointegration
The word “integration” refers to an integrated time series of order $d$, denoted by $I(d)$. According to Alexander et al. (Alexander, 2002), price, rate, and yield data can be assumed as $I(1)$ series, while returns (obtained by differencing the price) can be assumed as $I(0)$ series. The most important property of the $I(0)$ series that is relevant to statistical arbitrage is the following:

$I(0)$ series are **weak-sense stationary**.

Weak-sense stationarity implies that the mean and the variance of the time series are finite and do not change with time.

But wait, the $I(0)$ series is the returns: we cannot trade the returns! Only the price is tradable, yet the price is an $I(1)$ series, which are not stationary. We cannot make use of the stationary property of the $I(0)$ series by trading one asset.

What about two assets? According to the definition of cointegration (Alexander, 2002):

$x_{t} \text{ and } y_{t} \text{ are cointegrated}, \text{ if } x_{t} \text{ and } y_{t} \text{ are } I(1) \text{ series and }\exists \beta \text{ such that }z_{t} = x_{t} - \beta y_{t} \text{ is an }I(0) \text{ series}$

Cointegration allows us to construct a stationary time series from two asset price series, if only we can find the magic weight, or more formally, the cointegration coefficient $\beta$. Then we can apply a mean-reversion strategy to trade both assets at the same time weighted by \beta. There is no guarantee that such $\beta$ always exists, and you should look for other asset pairs if no such $\beta$ can be found.

**Correlation vs. Cointegration**
1. Correlation has no well-defined relationship with cointegration. Cointegrated series might have low correlation, and highly correlated series might not be cointegrated at all.
2. Correlation describes a short-term relationship between the returns.
3. Cointegration describes a long-term relationship between the prices.

#### Derivation of the cointegration coefficient $\beta$
The two workhorses of finding the cointegration coefficient \beta (or cointegration vector when there are more than 2 assets) are the Engle-Granger test (Engle, 1987) and the Johansen test.

**1. Engle-Granger Test**
The idea of Engle-Granger test is simple. We perform a linear regression between the two asset prices and check if the residual is stationary using the Augmented Dick-Fuller (ADF) test. If the residual is stationary, then the two asset prices are cointegrated. The cointegration coefficient is obtained as the coefficient of the regressor.

An immediate problem is in front of us. Which asset should we choose as the dependent variable? A feasible heuristic is that we run the linear regression twice using each asset as the dependent variable, respectively. The final $\beta$ would be the combination that yields a more significant ADF test.

**2. Johansen Test**
Johansen test uses the VECM to find the cointegration coefficient/vector $\beta$. The most important improvement of Johansen Test compared to Engle-Granger test is that it treats every asset as an independent variable. Johansen test also provides two test statistics, eigenvalue statistics and trace statistics, to determine if the asset prices are statistically significantly cointegrated.

In conclusion, Johansen test is a more versatile method of finding the cointegration coefficient/vector $\beta$ than the Engle-Granger test.
