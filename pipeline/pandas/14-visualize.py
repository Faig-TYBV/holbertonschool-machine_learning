#!/usr/bin/env python3
""" Visualize data """
import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__('2-from_file').from_file

# Load data
df = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',
    ','
)

# Drop unnecessary column
df = df.drop(columns="Weighted_Price")

# Rename Timestamp to Date (FIXED)
df = df.rename(columns={"Timestamp": "Date"})

# Convert timestamp to datetime (FIXED)
df["Date"] = pd.to_datetime(df["Date"], unit='s')

# Set Date as index
df = df.set_index("Date")

# Fill missing Close prices
df["Close"] = df["Close"].fillna(method="ffill")

# Fill missing Open, High, Low with Close
cols = dict.fromkeys(["High", "Low", "Open"], df["Close"])
df = df.fillna(value=cols)

# Fill missing volume with 0
cols = dict.fromkeys(["Volume_(BTC)", "Volume_(Currency)"], 0)
df = df.fillna(value=cols)

# Keep data from 2017 onwards
df = df[df.index >= '2017-01-01']

# Resample to daily data
daily_df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Print transformed DataFrame
print(daily_df)

# Plot OHLC prices
daily_df[['Open', 'High', 'Low', 'Close']].plot(figsize=(12, 6))
plt.title('BTC Price (Daily, 2017+)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()