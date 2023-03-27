# Statistical-Arbitrage

Here is my Pairs Trading Library. This implementation is supposed to be a free alternative to the Hudson & Thames Library for Statitical Arbitrage. I hope that this will lead to a profitable Pairs Trading Startegy. But for now this represents a proof of concept using pprice data of only a few tickers. 

### Euclidean Benchmark
We start with the simplest of all pairs trading strategies, which is calculating the Euclidean Distance between normalised price data of assets to form pairs based on the smallest distances. I will later combine this with a Minimum Profit Optimiation strategy that will tll us when to open our position and when to close. 

**STEP 1: Normalise Price Data**
$$P_{\text{norm}}=\frac{P-\text{min}(P)}{\text{max}(P)-\text{min}(P)}$$
