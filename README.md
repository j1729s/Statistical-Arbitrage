# Statistical-Arbitrage

Here is my Pairs Trading Library. This implementation is supposed to be a free alternative to the Hudson & Thames Library for Statitical Arbitrage. I hope that this will lead to a profitable Pairs Trading Startegy. But for now this represents a proof of concept using price data of only a few tickers. 

### Euclidean Benchmark
(Ref. Euclidean_Test.ipynb)
We start with the simplest of all pairs trading strategies, which is calculating the Euclidean Distance between normalised price data of assets to form pairs based on the smallest distances. I will later combine this with a Minimum Profit Optimiation strategy that will tll us when to open our position and when to close. 

**STEP 1: Normalise Price Data**
$$P_{\text{norm}}=\frac{P-\text{min}(P)}{\text{max}(P)-\text{min}(P)}$$
This is done in order to bring the price data of each asset to the same scale. 

**STEP 2: Calculate Sum of Squared Distances**
Using Euclidean squared distance on the normalized price time series, $n$ closest pairs of assets are picked.
$$\text{SSD}=\Sum^{N}_{1}{(P_{t}^{1}-P_{t}^{2})^{2}}$$
