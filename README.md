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

