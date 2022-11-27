import brian2 as b2
b2.prefs.codegen.target = 'numpy'
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import os
import requests as requests


num_epochs = 900 #1000 epochs
learning_rate = 0.001 #0.001 lr
input_size = 5 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out


class CustomGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def custom_gradient():
    """Straight Through Estimator surrogate gradient enclosed with a parameterized slope."""
    def inner(x):
        return CustomGradient.apply(x)
    return inner



spike_grad_custom = custom_gradient()

class SLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(SLSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.slstm = snn.SLSTM(input_size=input_size, hidden_size=hidden_size, spike_grad=spike_grad_custom) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer
        self.relu = nn.ReLU()
    
    def forward(self,x,num_steps=25):
        spk_rec = []
        mem_rec = []
        out_rec = []
        h_0 = Variable(torch.zeros(x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(x.size(0), self.hidden_size)) #internal state

        for _ in range(num_steps):
            spk, syn, mem = self.slstm(x.flatten(1), h_0, c_0)
            out = self.relu(mem)
            out = self.fc_1(out) #first Dense
            out = self.relu(out) #relu
            out = self.fc(out) #Final Output
            spk_rec.append(spk)
            mem_rec.append(mem)
            out_rec.append(out)
        return torch.stack(out_rec)


def get_snnTorch_preds(df, training=False):
    splt = int(df.shape[0] * 0.8)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    mm = MinMaxScaler()
    ss = StandardScaler()
    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y)

    X_train = X_ss[:splt, :]
    X_test = X_ss[splt:, :]
    y_train = y_mm[:splt, :]
    y_test = y_mm[splt:, :]

    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))
    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    #reshaping to rows, timestamps, features
    X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])
    slstm = SLSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])
    losses = []
    criterion = torch.nn.MSELoss()
    scriterion = torch.nn.MSELoss()

    if not training and os.path.exists('model/lstm.pt'):
        lstm.load_state_dict(torch.load('model/lstm.pt'))
        slstm.load_state_dict(torch.load('model/slstm.pt'))
    else:
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
        soptimizer = torch.optim.Adam(slstm.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        for epoch in range(num_epochs):
            outputs = lstm.forward(X_train_tensors_final) #forward pass
            optimizer.zero_grad()
            loss = criterion(outputs, y_train_tensors)
            loss.backward() #calculates the loss of the loss function
            optimizer.step()
            losses.append(loss.item())

            outs = slstm.forward(X_train_tensors_final) #forward pass
            soptimizer.zero_grad() #caluclate the gradient, manually setting to 0
            sloss = scriterion(outs.mean(axis=0), y_train_tensors)
            sloss.backward() #calculates the loss of the loss function
            soptimizer.step() #improve from loss, i.e backprop

            if not epoch % 100: print("Epoch: %d, loss: %1.5f, n-loss: %1.5f" % (epoch, loss.item(), sloss.item()))

    ytst = y_test_tensors.flatten().detach().numpy()
    yhat = lstm(X_test_tensors_final)
    syhat = slstm(X_test_tensors_final).mean(axis=0)
    loss = criterion(yhat, y_test_tensors)
    sloss = scriterion(syhat, y_test_tensors)
    yhat = yhat.flatten().detach().numpy()
    syhat = syhat.flatten().detach().numpy()
    pdf = pd.DataFrame({'ground_vol': ytst, 'lstm_vol': yhat, 'slstm_vol': syhat}, index=df.index.tolist()[splt:])

    torch.save(lstm.state_dict(), 'model/lstm.pt')
    torch.save(slstm.state_dict(), 'model/slstm.pt')
    return pdf, (loss,sloss)


def create_figure(pdf, col, title):
    ax = pdf.plot(figsize=(16,8))
    ax.set_ylabel('Volume')
    plt.sca(ax)
    if col=='ground_vol':
        plt.axvline(x = datetime.strptime('2020-09-24', '%Y-%m-%d').date(), c='r', linestyle='--') #size of the training set
    plt.title(title)
    plt.grid()
    plt.close()
    fig = ax.get_figure()
    return fig



'''

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


df_X_ss = ss.transform(df.iloc[:, :-1]) #old transformers
df_y_mm = mm.transform(df.iloc[:, -1:]) #old transformers
df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
dataY_plot = df_y_mm.data.numpy()
dataY_plot = mm.inverse_transform(dataY_plot)
train_predict = lstm(df_X_ss)#forward pass
data_predict = train_predict.data.numpy() #numpy conversion
data_predict = mm.inverse_transform(data_predict)
'''
