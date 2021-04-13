"""
Write a program that prints the numbers from 1 to 20. But for multiples of three, print 'Fizz'
instead of the number and for the multiples of five print 'Buzz'.
For numbers which are multiples of both three and five print 'FizzBuzz'.
"""

import pandas as pd
import streamlit as st
from PIL import Image
from datetime import date
from alpha_vantage.timeseries import TimeSeries
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
import numpy as np
import time
import yfinance as yf

# App title
st.markdown('''
# Stock Price App
Shown are the stock price data for query companies!
**Credits**
- App built by [Osato Osagie](https://www.linkedin.com/in/osato-osagie) (aka [HenchTechGuy](https://github.com/greggs25))
- Built in `Python` using `streamlit`,`yfinance`,`AlphaVantage`, `Scikit-learn`, `Tensorflow`,`pandas` and `datetime`
''')
st.write('---')


df_tickers = pd.read_csv('nasdaq_screener.csv')

sc = MinMaxScaler(feature_range=(0, 1))
# image = Image.open("/Users/aesthetic/Desktop/oo115/PycharmProjects/pythonProject/randomBotimg.png")
# st.image(image, use_column_width=True)

st.sidebar.header('Query parameters')

# present date
today = date.today()

# function to extract the list of functions
def getStockTickers():

    tickers = []
    # add empty symbol to clear dropdown
    tickers.append('')
    for i in df_tickers['Symbol']:
        tickers.append(i)

    return tickers



# function to obtain the users input
def get_input():
    start_date = st.sidebar.text_input("Start Date", "2015-01-01")
    end_date = st.sidebar.text_input("End Date", today)
    stock_symbol = st.sidebar.selectbox(
        'Select stock symbol',
        (getStockTickers()))
    num_days = st.sidebar.selectbox(
        'Select Number of Days',
        (1, 2, 3, 4, 5, 6, 7))
    model = st.sidebar.selectbox(
        'Select Model',
        (['Ridge Regression', 'Random Forest', 'LSTM']))
    return start_date, end_date, stock_symbol, num_days, model


# sym = 'AAPL'

# funtion to obtain data from alpha vantage API
def stock_data(sym, date_of_data, end_date):
    try:
        ts = TimeSeries(key='SUXOFAIGXM6HEP9Y', output_format='pandas')
        data, meta_data = ts.get_daily_adjusted(symbol=sym, outputsize='full')
        data_date_changed = data[end_date:date_of_data]
        data_date_changed['Ticker'] = sym
        data_date_changed.sort_index(ascending=True, inplace=True)

        return data_date_changed
    except ValueError as ve:
        st.write("Sorry Alpha vantage has a limit of 5 calls per minute, please wait...")
        progress_bar = st.progress(0)
        progress_bar.progress(1)
        time.sleep(60)
        for i in range(99):
            # Update progress bar.
            progress_bar.progress(i + 1)
        st.experimental_rerun()


# funtion for interative plots
def interactive_plot(df, title):
    fig = px.line(title=title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['date'], y=df[i], name=i)
    st.write(fig)


# get users input
start_date, end_date, symbol, num_days, model = get_input()




# function to check if ticker is blank
if symbol == '':
    symbol = 'A'




# get the stock data
data_dated = stock_data(symbol, start_date, end_date)

# Get ticker data for logo
tickerData = yf.Ticker(symbol)

# Ticker information
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)


# display the close price
st.header( " Close Price\n")
st.line_chart(data_dated['4. close'])


def show_corr():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data_dated.corr(), center=0, cmap='Blues', annot=True)
    ax.set_title('Multi-Collinearity of Car Attributes')
    st.write(fig)


st.header(" Data Correlation")
show_corr()

# get statistics on the data
st.header(" Data Statistics")
st.write(data_dated.describe())


# Function to return the input/output (target) data for AI/ML Model
# Note that our goal is to predict the future stock price
# Target stock price today will be tomorrow's price
def trading_window(data, days):
    # 1 day window
    #   n = 1

    # Create a column containing the prices for the next 1 days
    data['Target'] = data[['4. close']].shift(-days)

    # return the new dataset
    return data


# calculate the change in price from the previous day
data_dated['change_in_price'] = data_dated['4. close'].diff()


# calculating the Relative Strength Index

def calc_RSI(data_dated):
    # Calculate the 14 day RSI
    n = 14

    # First make a copy of the data frame twice
    up_df, down_df = data_dated[['Ticker', 'change_in_price']].copy(), data_dated[['Ticker', 'change_in_price']].copy()

    # For up days, if the change is less than 0 set to 0.
    up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0

    # For down days, if the change is greater than 0 set to 0.
    down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0

    # We need change in price to be absolute.
    down_df['change_in_price'] = down_df['change_in_price'].abs()

    # Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
    ewma_up = up_df.groupby('Ticker')['change_in_price'].transform(lambda x: x.ewm(span=n).mean())
    ewma_down = down_df.groupby('Ticker')['change_in_price'].transform(lambda x: x.ewm(span=n).mean())

    # Calculate the Relative Strength
    relative_strength = ewma_up / ewma_down

    # Calculate the Relative Strength Index
    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

    # Add the info to the data frame.
    data_dated['down_days'] = down_df['change_in_price']
    data_dated['up_days'] = up_df['change_in_price']
    data_dated['RSI'] = relative_strength_index


# Claclulting the Stochastic Oscillator
def stochastic_Oscillator(data_dated):
    # Calculate the Stochastic Oscillator
    n = 14

    # Make a copy of the high and low column.
    low_14, high_14 = data_dated[['Ticker', '3. low']].copy(), data_dated[['Ticker', '2. high']].copy()

    # Group by symbol, then apply the rolling function and grab the Min and Max.
    low_14 = low_14.groupby('Ticker')['3. low'].transform(lambda x: x.rolling(window=n).min())
    high_14 = high_14.groupby('Ticker')['2. high'].transform(lambda x: x.rolling(window=n).max())

    # Calculate the Stochastic Oscillator.
    k_percent = 100 * ((data_dated['4. close'] - low_14) / (high_14 - low_14))

    # Add the info to the data frame.
    data_dated['low_14'] = low_14
    data_dated['high_14'] = high_14
    data_dated['k_percent'] = k_percent


# calculating williams R%
def calc_williams_r(data_dated):
    # Calculate the Williams %R
    n = 14
    # Make a copy of the high and low column.
    low_14, high_14 = data_dated[['Ticker', '3. low']].copy(), data_dated[['Ticker', '2. high']].copy()
    # Group by symbol, then apply the rolling function and grab the Min and Max.
    low_14 = low_14.groupby('Ticker')['3. low'].transform(lambda x: x.rolling(window=n).min())
    high_14 = high_14.groupby('Ticker')['2. high'].transform(lambda x: x.rolling(window=n).max())
    # Calculate William %R indicator.
    r_percent = ((high_14 - data_dated['4. close']) / (high_14 - low_14)) * - 100
    # Add the info to the data frame.
    data_dated['r_percent'] = r_percent


def calc_macd(data_dated):
    # Calculate the MACD
    ema_26 = data_dated.groupby('Ticker')['4. close'].transform(lambda x: x.ewm(span=26).mean())
    ema_12 = data_dated.groupby('Ticker')['4. close'].transform(lambda x: x.ewm(span=12).mean())
    macd = ema_12 - ema_26

    # Calculate the EMA
    ema_9_macd = macd.ewm(span=9).mean()

    # Store the data in the data frame.
    data_dated['MACD'] = macd
    data_dated['MACD_EMA'] = ema_9_macd


def calc_price_rate_of_change(data_dated):
    # Calculate the Price Rate of Change
    n = 9

    # Calculate the Rate of Change in the Price, and store it in the Data Frame.
    data_dated['Price_Rate_Of_Change'] = data_dated.groupby('Ticker')['4. close'].transform(
        lambda x: x.pct_change(periods=n))


starting_date = '2015-01-01'
pd.options.mode.chained_assignment = None  # default='warn'

# Evaluation metrics
RMSE = []
Rsquared = []
Mae = []


# Building ridge regression model
def pricePrediction_LR(symbol, days, starting_date, end_date):
    # check if symbol is blank
    if symbol == '':
        symbol = 'A'
    #     obtain stock data
    stock_df = stock_data(symbol, starting_date, end_date)

    #     obtaining technical indicators
    stochastic_Oscillator(stock_df)
    calc_williams_r(stock_df)
    calc_macd(stock_df)
    calc_price_rate_of_change(stock_df)

    #     set the trading window we are trying to predict
    stock_df_targeted = trading_window(stock_df, days)
    # #     remove the last column of the data as it will be null
    #     stock_df_targeted= stock_df_targeted[:-1]

    stock_df_targeted.reset_index(inplace=True)
    stock_df_targeted = stock_df_targeted.dropna()

    stock_df_targeted_scaled = stock_df_targeted
    stock_df_targeted_scaled.head(10)
    stock_df_targeted_scaled.drop(
        ['Ticker', '4. close', '7. dividend amount', '3. low', '5. adjusted close', '6. volume', '8. split coefficient',
         'low_14', 'high_14', 'MACD_EMA'], axis=1, inplace=True)

    stock_df_targeted_scaled = sc.fit_transform(stock_df_targeted_scaled.drop(columns=['date']))

    # Creating Feature and Target
    X = stock_df_targeted_scaled[:, :6]
    y = stock_df_targeted_scaled[:, 6:]

    split = int(0.65 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    # show_plot(X_train, 'Training Data')
    # show_plot(X_test, 'Testing Data')

    regression_model = Ridge()
    regression_model.fit(X_train, y_train)

    lr_accuracy = regression_model.score(X_test, y_test)
    predicted_prices = regression_model.predict(X)

    Predicted = []
    for i in predicted_prices:
        Predicted.append(i[0])

    close = []
    for i in stock_df_targeted_scaled:
        close.append(i[0])

    df_predicted = stock_df_targeted[['date']]
    df_predicted['Close'] = close
    df_predicted['Prediction'] = Predicted

    RMSE.append(math.sqrt(mean_squared_error(y, predicted_prices)))
    Rsquared.append(r2_score(y, predicted_prices))
    Mae.append(mean_absolute_error(y, predicted_prices))

    interactive_plot(df_predicted, "Original Vs. Prediction")


# randomForest Model
def pricePrediction_RandomForest(symbol, days, start_date, end_date):
    p = 0
    mse = []
    rmse = []
    rsquared = []
    mae = []

    #     obtain stock data
    stock_df = stock_data(symbol, start_date, end_date)

    #     obtaining technical indicators
    stochastic_Oscillator(stock_df)
    calc_williams_r(stock_df)
    calc_macd(stock_df)
    calc_price_rate_of_change(stock_df)

    #     set the trading window we are trying to predict
    stock_df_targeted = trading_window(stock_df, days)

    stock_df_targeted.reset_index(inplace=True)
    stock_df_targeted = stock_df_targeted.dropna()

    stock_df_targeted_scaled = stock_df_targeted
    #     stock_df_targeted_scaled.head(10)
    stock_df_targeted_scaled.drop(
        ['Ticker', '4. close', '7. dividend amount', '3. low', '5. adjusted close', '6. volume', '8. split coefficient',
         'low_14', 'high_14', 'MACD_EMA'], axis=1, inplace=True)

    stock_df_targeted_scaled = sc.fit_transform(stock_df_targeted_scaled.drop(columns=['date']))

    #     # Creating Feature and Target
    X = stock_df_targeted_scaled[:, :6]
    y = stock_df_targeted_scaled[:, 6:]

    split = int(0.65 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    # show_plot(X_train, 'Training Data')
    # show_plot(X_test, 'Testing Data')

    rf = RandomForestRegressor()

    rf.fit(X_train, y_train.ravel())
    pred_rf = rf.predict(X)

    Predicted = []
    for i in pred_rf:
        Predicted.append(i)

    close = []
    for i in stock_df_targeted_scaled:
        close.append(i[0])

    df_predicted = stock_df_targeted[['date']]
    df_predicted['Close'] = close
    df_predicted['Prediction'] = Predicted

    RMSE.append(math.sqrt(mean_squared_error(y, pred_rf)))
    Rsquared.append(r2_score(y, pred_rf))
    Mae.append(mean_absolute_error(y, pred_rf))

    interactive_plot(df_predicted, "Original Vs. Prediction for ")


# LSTM model
def pricePrediction_LSTM(symbol, days, start_date, end_date):
    #     obtain stock data

    stock_df = stock_data(symbol, start_date, end_date)

    #     obtaining technical indicators
    stochastic_Oscillator(stock_df)
    calc_williams_r(stock_df)
    calc_macd(stock_df)
    calc_price_rate_of_change(stock_df)

    stock_df.reset_index(inplace=True)

    #     set the trading window we are trying to predict
    stock_df_targeted = trading_window(stock_df, days)
    stock_df_targeted.drop(
        ['Ticker', '4. close', '7. dividend amount', '3. low', '5. adjusted close', '6. volume', '8. split coefficient',
         'low_14', 'high_14', 'MACD_EMA'], axis=1, inplace=True)
    stock_df_targeted.dropna(inplace=True)
    training_data_X = stock_df_targeted.iloc[:, 1:6].values
    training_data_y = stock_df_targeted.iloc[:, 6:].values

    stock_df_targeted_scaled = sc.fit_transform(stock_df_targeted.drop(columns=['date']))

    X = sc.fit_transform(training_data_X)
    y = sc.fit_transform(training_data_y)

    # Convert the data into array format
    X = np.asarray(X)
    y = np.asarray(y)

    # Split the data
    split = int(0.7 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    # Reshape the 1D arrays to 3D arrays to feed in the model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Create the model
    inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = keras.layers.LSTM(150, return_sequences=True)(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(150, return_sequences=True)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(150)(x)
    outputs = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss="mse", metrics=['mean_squared_error', 'mae'])
    model.summary()

    # Trai the model
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    predicted = model.predict(X)

    test_predicted = []

    for i in predicted:
        test_predicted.append(i[0])

    close = []
    for i in stock_df_targeted_scaled:
        close.append(i[0])

    df_predicted = stock_df_targeted[['date']]
    df_predicted['Close'] = close
    df_predicted['Prediction'] = predicted

    #     interactive_plot(df_predicted, "Original Vs. Prediction for " )
    scores = model.evaluate(X, y, verbose=0)

    RMSE.append(math.sqrt(mean_squared_error(y, predicted)))
    Rsquared.append(r2_score(y, predicted))
    Mae.append(mean_absolute_error(y, predicted))

    # Plot the data
    interactive_plot(df_predicted, "Original Vs Prediction")


# 'Ridge Regression', 'Random Forest', 'LSTM'
if model == 'Ridge Regression':
    st.header("Ridge Regression Model for " + str(num_days) + " Day(s)")
    # Pretend we're doing some computation that takes time.
    progress_bar = st.progress(0)

    for i in range(100):
        # Update progress bar.
        progress_bar.progress(i + 1)

    pricePrediction_LR(symbol, num_days, start_date, end_date)
    progress_bar.balloons();
elif model == 'Random Forest':
    st.header("Random Forest model for " + str(num_days) + " Day(s)")
    pricePrediction_RandomForest(symbol, num_days, start_date, end_date)
elif model == 'LSTM':

    st.header("LSTM Model for " + str(num_days) + " Day(s)")
    pricePrediction_LSTM(symbol, num_days, start_date, end_date)

st.header("Evaluation metrics")
# add evaluation metrics to dataframe
evaluation_metrics = {'Rsquared': Rsquared, 'Mae': Mae, 'RMSE': RMSE}

# add eval metrics to table
chart = st.table(evaluation_metrics)

st.sidebar.button("Run")
