import ccxt

import requests

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from datetime import datetime, timedelta

# Configure the exchange and get price data
exchange = ccxt.binance()

symbol = 'BTC/EUR'

timeframe = '30m'

# Set the time period for historical data
start_date = datetime.now() - timedelta(days=365)

end_date = datetime.now()

# Download historical data for the specified time period
ohlcv_historical = exchange.fetch_ohlcv(symbol, timeframe, since=int(start_date.timestamp()*1000), limit=None)

df_historical = pd.DataFrame(ohlcv_historical, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

df_historical['timestamp'] = pd.to_datetime(df_historical['timestamp'], unit='ms')

df_historical.set_index('timestamp', inplace=True)

# Train the linear regression model
model = LinearRegression()

X = pd.DataFrame(df_historical['close'].values[:-1], columns=['yesterday_close'])

y = pd.DataFrame(df_historical['close'].values[1:], columns=['today_close'])

model.fit(X, y)

# Get the most recent closing price
ohlcv_recent = exchange.fetch_ohlcv(symbol, timeframe, limit=2)

df_recent = pd.DataFrame(ohlcv_recent, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'], unit='ms')

df_recent.set_index('timestamp', inplace=True)

recent_price = df_recent['close'].iloc[-1]

# Initialize the chart figure
fig, ax = plt.subplots()

# Get the current price
url = 'https://api.binance.com/api/v3/ticker/price'

params = {'symbol': symbol.replace('/', '')}

response = requests.get(url, params=params)

json_response = response.json()

realtime_price = float(json_response['price'])

# Make a prediction for the next 30 minutes
prediction = model.predict([[recent_price]])

# Print the results
print(f"Current closing price: {realtime_price:.2f} EUR")

print(f"Prediction for the next 30 minutes: {prediction[0][0]:.2f} EUR")

# Update the current price in the terminal title
ax.set_title(f"BTC/EUR - Current Price: {realtime_price:.2f} EUR")

# Plot historical data and current price
ax.plot(df_historical.index, df_historical['close'], label='Closing Price')

ax.plot(df_recent.index, df_recent['close'], label='Current Price', linewidth=2)

# Add legend and show the chart
ax.legend()

plt.show()
