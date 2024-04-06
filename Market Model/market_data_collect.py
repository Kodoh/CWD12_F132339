# Script to get Yahoo Finance crypto market data 
import yfinance as yf
import shutil
import os
import time

tickers = ["BTC-GBP", "ETH-GBP", "ADA-GBP", "XRP-GBP", "XLM-GBP", "LTC-GBP", "BNB-GBP", "DOGE-GBP"]

shutil.rmtree("/Daily_Stock_Report")
os.mkdir("/Daily_Stock_Report")
Amount_of_API_Calls = 0
Stock_Failure = 0
Stocks_Not_Imported = 0
i = 0
while (i < len(tickers)) and (Amount_of_API_Calls < 1800):
    try:
        stock = tickers[i]
        temp = yf.Ticker(str(stock))
        Hist_data = temp.history(period="max")
        Hist_data.to_csv("/Daily_Stock_Report/Stocks" + stock + ".csv")
        time.sleep(2)
        Amount_of_API_Calls += 1
        Stock_Failure = 0
        i += 1
    except ValueError:
        print("Yahoo Finance Backend Error, Attempting to Fix")
        if Stock_Failure > 5:
            i += 1
            Stocks_Not_Imported += 1
        Amount_of_API_Calls += 1
        Stock_Failure += 1
print("The amount of stocks we successfully imported: " + str(i - Stocks_Not_Imported))


