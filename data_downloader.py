import yfinance as yf

DATA_DIR = "data"

def download_stock(stock_name: str):
    stock = yf.Ticker(stock_name)
    stock_hist = stock.history(period="10y", interval='1d')
    
    total_time = len(stock_hist)
    index = int(0.9*total_time)
    
    train_data = stock_hist[:index]
    test_data = stock_hist[index:]
    
    train_path = f"{DATA_DIR}/{stock_name}_train.csv"
    test_path = f"{DATA_DIR}/{stock_name}_test.csv"
    train_data.to_csv(train_path)
    test_data.to_csv(test_path)



if __name__ == "__main__":
    stock_ticker_symbols = ["MSFT", "GOOG", "META", "MANU", "PSX"]
    for stock_name in stock_ticker_symbols:
        download_stock(stock_name)