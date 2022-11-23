import brian2 as b2
b2.prefs.codegen.target = 'numpy'
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


#Geometric Brownian motion
GBM = 'dX/dt = (mu-0.5*sigma**2)*dt*Hz/second + sigma*dt**-0.5*xi : 1'

#Get ticker forecasts
def get_brian2_preds(prices, end_date, pred_end_date, seed = 12347):
    train_set = prices.loc[:end_date]
    test_set = prices.loc[end_date:pred_end_date]
    daily_returns = ((train_set / train_set.shift(1)) - 1)[1:]

    N = pd.date_range(start=pd.to_datetime(end_date,format="%Y-%m-%d") + pd.Timedelta('1 days'),
        end=pd.to_datetime(pred_end_date,format="%Y-%m-%d")).to_series().map(lambda x: 1 if x.isoweekday() in range(1, 6) else 0).sum()
    S_0 = train_set[-1]
    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns)
    np.random.seed(seed)
    
    G = b2.NeuronGroup(N, GBM,  method='euler')
    mon = b2.StateMonitor(G, 'X', record=True)
    net = b2.Network(G, mon)
    net.run(1000*b2.ms)
    xx = np.array(mon.X)
    dS = np.insert(xx, 0, 0, axis=1)
    S = np.cumsum(dS, axis=1)
    S = S_0*np.exp(S)
    S_max = [S[:, i].max() for i in range(0, int(N))]
    S_min = [S[:, i].min() for i in range(0, int(N))]
    S_pred = .5 * np.array(S_max) + .5 * np.array(S_min)
    final_df = pd.DataFrame(data=[test_set.reset_index()['Adj Close'], S_pred],index=['real', 'pred']).T
    final_df.index = test_set.index
    mse = 1/len(final_df) * np.mean((final_df['pred'] - final_df['real']) ** 2)
    return final_df, mse



#scrapes yahoo finance
def scrape(stock_name, start_date, pred_end_date, interval):
    prices = yf.download(tickers=stock_name, start=start_date, end=pred_end_date, interval=interval).round(3)
    return prices


#creates plot of ticker history
def create_figure(pdf, col):
    scale=5
    ax = pdf.plot(figsize=(10, 6))
    ax.set_ylim(pdf[col].mean() - scale*pdf[col].std(), pdf[col].mean() + scale*pdf[col].std())
    ax.set_ylabel('Price $ (USD)')
    plt.sca(ax)
    plt.grid()
    fig = ax.get_figure()
    return fig