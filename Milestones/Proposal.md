# Adaptive Regime-Switching Investment Strategy 
### Team: Jack Bray, Alex Shields, Getchell Gibbons 
## Overview: 
We aim to build an adaptive investment strategy that detects and responds to changing market 
regimes—helping investors improve returns and manage risk during volatile periods. Traditional 
static portfolios often fail during regime shifts (bull to bear markets, etc), so our approach uses 
machine learning to identify and adapt to these changes. 
## Data & Features: 
We’ll use yfinance to gather 10–15 years of daily data for key ETFs across major asset classes 
(SPY, QQQ, TLT, AGG, GLD, EEM, VNQ). From prices, we’ll compute returns, volatilities, 
correlations, and technical indicators (moving averages, momentum, breadth, etc.) to capture 
market dynamics. We will perform data exploration to find interesting patterns, such as 
heatmaps, KDE’s, etc.  
## Hidden Markov Model (HMM): 
Using hmmlearn, we’ll train an HMM with 2–4 hidden states to uncover distinct market regimes 
(bull, bear, sideways, etc.). The model will learn transition probabilities and identify when each 
regime occurs. Visualization will show how detected regimes align with market performance. 
## Regime-Based Portfolio Optimization: 
For each regime, we’ll apply mean-variance optimization (via cvxpy) to compute optimal portfolio 
weights based on regime-specific returns and covariance matrices. This allows allocations to 
shift as market behavior changes. 
##Dynamic Strategy & Backtesting: 
We’ll build a rolling backtest that updates the HMM and rebalances portfolios monthly, using 
only past data to avoid look-ahead bias. Transaction costs, position limits, and rebalancing 
thresholds will ensure realism. 
## Evaluation: 
We’ll compare our regime-aware strategy against benchmarks like a 60/40 portfolio, 
equal-weighted assets, and S&P 500 buy-and-hold using metrics such as Sharpe ratio, Sortino 
ratio, and max drawdown. We will plot returns over time as well as color prices by regime and 
compare them to what we expected. 
## Goal: 
Demonstrate that a regime-switching, data-driven approach can outperform static strategies in 
real-world conditions.