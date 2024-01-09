import yfinance as yf
import streamlit as st
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn import metrics

st.write("""
# Simple Stock Price App

Shown are the stock **closing price** and ***volume*** of Google!

""")

# Define a list of ticker symbols for 200 stocks (you can modify this list accordingly)
ticker_symbols = ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'INTC', 'IBM',
                   'NFLX', 'CSCO', 'V', 'PYPL', 'GS', 'JPM', 'WMT', 'CVX', 'XOM', 'BA',
                   'DIS', 'KO', 'PEP', 'MCD', 'NKE', 'WBA', 'PG', 'UNH', 'VZ', 'HD'
                   # Add more ticker symbols as needed...
                   # Repeat to have a total of 200 stocks
                   ]

# Get data on the selected tickers
data = pd.DataFrame()
for ticker_symbol in ticker_symbols:
    ticker_data = yf.Ticker(ticker_symbol).history(period='1d', start='2000-5-31', end='2020-5-31')
    data[ticker_symbol] = ticker_data['Close']

st.write("""
## Closing Prices
""")
st.line_chart(data)

# Predict the closing price of Google based on the closing prices of other stocks
google_ticker = 'GOOGL'
X = data.drop(google_ticker, axis=1)  # Features (closing prices of other stocks)
y = data[google_ticker]  # Target (closing price of Google stock)

# Use time-based split (TimeSeriesSplit) to split the data
tscv = TimeSeriesSplit(n_splits=40)  # You can adjust the number of splits as needed

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Train the XGBoost model with good hyperparameters
model = XGBRegressor(
    n_estimators=1000,  # Number of boosting rounds
    learning_rate=0.01,  # Step size shrinkage used in each boosting step
    max_depth=5,  # Maximum depth of a tree
    subsample=0.8,  # Fraction of samples used for fitting the individual base learners
    colsample_bytree=0.8,  # Fraction of features used for fitting the individual base learners
    random_state=42,
)

model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Display the model's performance metrics
st.write("""
## XGBoost Model Performance
""")
st.write(f'Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}')
st.write(f'Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}')
st.write(f'Root Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred, squared=False)}')

# Predict and display the closing prices of Google based on the other stocks
predicted_google_prices = model.predict(X)
st.write("""
## Predicted Closing Prices of Google (based on Other Stocks)
""")
st.line_chart(pd.Series(predicted_google_prices, index=data.index))



##############
