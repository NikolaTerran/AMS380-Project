# Group 1: Neural Networks and Asset Price Prediction
# Antonio Frigo, 112747350
# Justin Willson, 112649988
# Jeffrey Zheng, 113821367
# Tianrun Liu, 112838591

import yfinance as yf

DATA_DIR = "data"

def download_stock(stock_name: str):
    stock = yf.Ticker(stock_name)
    stock_hist = stock.history(period="10y", interval='1d')
    
    total_time = len(stock_hist)
    # val_id = int(0.8*total_time)
    index = int(0.9*total_time)
    
    train_data = stock_hist[:index]
    # val_data = stock_hist[val_id:index]
    test_data = stock_hist[index:]
    
    train_path = f"{DATA_DIR}/{stock_name}_train.csv"
    # val_path = f"{DATA_DIR}/{stock_name}_val.csv"
    test_path = f"{DATA_DIR}/{stock_name}_test.csv"
    train_data.to_csv(train_path)
    # val_data.to_csv(val_path)
    test_data.to_csv(test_path)



if __name__ == "__main__":
    stock_ticker_symbols = ["MSFT", "GOOG", "META", "MANU", "PSX"]
    for stock_name in stock_ticker_symbols:
        download_stock(stock_name)
