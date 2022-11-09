# use this script to generate csv file for training/testing
# doc: https://pypi.org/project/yfinance/
import yfinance as yf
from sklearn.model_selection import train_test_split

msft = yf.Ticker("MSFT")

# get stock info
msft.info

# start with 10 year microsoft data
ms_hist = msft.history(period="10y", interval='1d')

# split to 9 year train and 1 year test
ms_train, ms_test = train_test_split(ms_hist, test_size=0.1, shuffle=False)

ms_train.to_csv("data/ms_train",index=True)
ms_test.to_csv("data/ms_test",index=True)