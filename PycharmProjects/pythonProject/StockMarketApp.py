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

st.write(""" #Stock Market Web Application
**Visually** show data on a stock! Date range from Jan 2, 2020 - Aug 4, 2020
""")

image= Image.open("/Users/aesthetic/Desktop/oo115/PycharmProjects/pythonProject/randomBotimg.png")
st.image(image, use_column_width=True)

st.sidebar.header('User Input')


# present date
today = date.today()

print("Today's date:", today)

# function to obtain the users input
def get_input():
    start_date = st.sidebar.text_input("Start Date", "2015-01-01")
    end_date = st.sidebar.text_input("End Date", today)
    stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
    return start_date, end_date, stock_symbol

# sym = 'AAPL'

# funtion to obtain data from alpha vantage API
def stock_data(sym, date_of_data, end_date):
    ts = TimeSeries(key='SUXOFAIGXM6HEP9Y', output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol=sym, outputsize='full')
    data_date_changed = data[end_date:date_of_data]
    data_date_changed['Ticker'] = sym
    return data_date_changed

# funtion for interative plots
def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['date'], y = df[i], name = i)
  fig.show()

# get users input
start, end,symbol= get_input()
# get the data
data_dated= stock_data(symbol,start,end)

# st.write(data_dated)

# display the close price
st.header(symbol+" Close Price\n")
st.line_chart(data_dated['4. close'])

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(data_dated.corr(), center=0, cmap='Blues' , annot=True)
ax.set_title('Multi-Collinearity of Car Attributes')

st.write(fig)

#get statistics on the data
st.header(symbol+" Data Statistics")
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
    up_df, down_df = data_dated[['Ticker','change_in_price']].copy(), data_dated[['Ticker','change_in_price']].copy()

    # For up days, if the change is less than 0 set to 0.
    up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0

    # For down days, if the change is greater than 0 set to 0.
    down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0

    # We need change in price to be absolute.
    down_df['change_in_price'] = down_df['change_in_price'].abs()

    # Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
    ewma_up = up_df.groupby('Ticker')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
    ewma_down = down_df.groupby('Ticker')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())

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
    low_14, high_14 = data_dated[['Ticker','3. low']].copy(), data_dated[['Ticker','2. high']].copy()

    # Group by symbol, then apply the rolling function and grab the Min and Max.
    low_14 = low_14.groupby('Ticker')['3. low'].transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.groupby('Ticker')['2. high'].transform(lambda x: x.rolling(window = n).max())

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
    low_14, high_14 = data_dated[['Ticker','3. low']].copy(), data_dated[['Ticker','2. high']].copy()
    # Group by symbol, then apply the rolling function and grab the Min and Max.
    low_14 = low_14.groupby('Ticker')['3. low'].transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.groupby('Ticker')['2. high'].transform(lambda x: x.rolling(window = n).max())
    # Calculate William %R indicator.
    r_percent = ((high_14 - data_dated['4. close']) / (high_14 - low_14)) * - 100
    # Add the info to the data frame.
    data_dated['r_percent'] = r_percent


def calc_macd(data_dated):
    # Calculate the MACD
    ema_26 = data_dated.groupby('Ticker')['4. close'].transform(lambda x: x.ewm(span = 26).mean())
    ema_12 = data_dated.groupby('Ticker')['4. close'].transform(lambda x: x.ewm(span = 12).mean())
    macd = ema_12 - ema_26

    # Calculate the EMA
    ema_9_macd = macd.ewm(span = 9).mean()

    # Store the data in the data frame.
    data_dated['MACD'] = macd
    data_dated['MACD_EMA'] = ema_9_macd


def calc_price_rate_of_change(data_dated):
    # Calculate the Price Rate of Change
    n = 9

    # Calculate the Rate of Change in the Price, and store it in the Data Frame.
    data_dated['Price_Rate_Of_Change'] = data_dated.groupby('Ticker')['4. close'].transform(lambda x: x.pct_change(periods = n))




# allow user to select model
option = st.sidebar.selectbox(
  'Select Model?',
 ('Ridge Regression', 'Random Forest', 'LSTM'))




st.sidebar.button("Run")