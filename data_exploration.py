import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

symbols = ["MSFT", "GOOG", "META", "MANU", "PSX"] #list of stocks used
x_locs = [0 for i in range(len(symbols))]    #lists of the x and y coordinates for the text in the plots
y_locs = [2.5, 2.5, 2.0, 3.75, 2.25]

for i, symbol in enumerate(symbols):
    train_data = pd.read_csv(f"data/{symbol}_train.csv") #load training and test data
    test_data = pd.read_csv(f"data/{symbol}_test.csv")
    
    num_days = len(train_data)+len(test_data)    #create list of days
    days = [i for i in range(num_days)]
    
    train_mean = np.mean(train_data['Close'])  #calculate mean and sd of train data
    train_sd = np.std(train_data['Close'])
    
    train_data_norm = (train_data['Close']-train_mean)/train_sd #normalize with train data mean and sd
    test_data_norm = (test_data['Close']-train_mean)/train_sd
    
    fig, ax = plt.subplots(figsize=(12,7), tight_layout=True)   #plot data
    ax.plot(days[:len(train_data)], train_data_norm, label=f'{symbol} train data')
    ax.plot(days[len(train_data):], test_data_norm, label=f'{symbol} test data')
    ax.text(x_locs[i], y_locs[i], f"n={num_days}", fontsize=14)
    ax.text(x_locs[i], y_locs[i]-0.3, f"original train mean={np.round(train_mean, 2)}", fontsize=14)
    ax.text(x_locs[i], y_locs[i]-0.6, f"original train sd={np.round(train_sd, 2)}", fontsize=14)
    ax.set_title(f'{symbol}', fontsize=22)
    ax.set_ylabel('Normalized Price', fontsize=18)
    ax.set_xlabel('Days', fontsize=18)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=14)
    plt.savefig(f"plots/{symbol}_data_exploration.png", dpi=300, bbox_inches='tight')                                                      