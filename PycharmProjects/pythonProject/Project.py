#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install yahoo-finance


# In[2]:


# pip install alpha_vantage pandas


# In[3]:


# pip install yfinance


# In[4]:


#pip install statsmodels


# In[5]:


import nltk
import pandas as pd
import streamlit as st
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
from sklearn.model_selection import KFold
import time
import yfinance as yf
import nltk
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize, sent_tokenize
import plotly.graph_objects as go
nltk.download('punkt')


# In[6]:


# App title and Credits
st.markdown('''
# Stock Price App
Shown are the stock price data for query companies!
**Credits**
- App built by [Osato Osagie](https://www.linkedin.com/in/osato-osagie) (aka [HenchTechGuy](https://github.com/greggs25))
- Built in `Python` using `Streamlit`,`yfinance`,`AlphaVantage`, `Scikit-learn`, `Tensorflow`,`Pandas`, `Plotly`, and `Seaborn`
''')
# header
st.write('---')


# In[7]:

# stock tickers for dropdown list
df_tickers = pd.read_csv('nasdaq_screener.csv')


# In[8]:

# scaler for feature variables
sc = MinMaxScaler(feature_range=(0, 1))
# scaler for dependent variables
y_sc = MinMaxScaler(feature_range=(0, 1))

# In[9]:

# side bar title
st.sidebar.header('Query parameters')

# present date
today = date.today()

# list of all buys from trading algorithms
buys = []
# list of all the sells made from trading algorithm
sells = []
# threshold for trading algorithm
thresh = 0.2



# function to extract the list of tickers
def getStockTickers():

    tickers = []
    # add empty symbol to clear dropdown
    tickers.append('')
    for i in df_tickers['Symbol']:
        tickers.append(i)

    return tickers


# In[11]:

# function to obtain the users input
def get_input():
#     the starting period of the data we obtain from the Alpha Vantage API.
    start_date = st.sidebar.text_input("Start Date", "2015-01-01")
#     the end period of the data obtained from the Alpha Vantage API.
    end_date = st.sidebar.text_input("End Date", today)
#    a string of characters (usually letters) that represents publicly traded companies on an exchange
    stock_symbol = st.sidebar.selectbox(
        'Select Stock Symbol',
        (getStockTickers()))
    num_days = st.sidebar.number_input("Number of day(s)", 1)
    investment_amount = st.sidebar.number_input("Investment Amount (££)", 10)
    model = st.sidebar.selectbox(
        'Select Model',
        (['Ridge Regression', 'Random Forest', 'LSTM']))
    return start_date, end_date, stock_symbol, num_days, model, investment_amount


# In[12]:


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


# In[13]:


# funtion for interative plots
def interactive_plot(df, title):
    fig = px.line(title=title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['date'], y=df[i], name=i)
    fig.update_layout(

        xaxis_title="Year",
        yaxis_title="Price",

    )
    st.write(fig)


# In[14]:


# function for plotting the closing price
def interactive_plot_closingPrice(df, title):
    fig = px.line()
    fig.add_scatter(x=df.index, y=df['4. close'], name="Close")
    st.write(fig)


# In[15]:


# get users input
start_date, end_date, symbol, num_days, model, investment_amount = get_input()


# In[16]:




# function to check if ticker is blank
if symbol == '':
    symbol = 'A'


# In[17]:


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


# In[18]:


def summarise_text(string_summary):
    #  text to sum up
    message = string_summary

    # Text tokenization
    haltWords = set(sw.words("english"))
    phrases = word_tokenize(message)

    # Making a frequency distribution table to record each word's score
    frequencyTable  = dict()
    for word in phrases:
        word = word.lower()
        if word in haltWords:
            continue
        if word in frequencyTable:
            frequencyTable[word] += 1
        else:
            frequencyTable[word] = 1

    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(message)
    sentenceWorthiness = dict()

    for i in sentences:
        for word, freq in frequencyTable.items():
            if word in i.lower():
                if i in sentenceWorthiness:
                    sentenceWorthiness[i] += freq
                else:
                    sentenceWorthiness[i] = freq

    totalOfValues  = 0
    for i in sentenceWorthiness:
        totalOfValues += sentenceWorthiness[i]

    # Average value of a sentence from the original text

    average = int(totalOfValues / len(sentenceWorthiness))

    # Storing sentences into our summary.
    textsummary = ''
    for sentence in sentences:
        if (sentence in sentenceWorthiness) and (sentenceWorthiness[sentence] > (1.2 * average)):
            textsummary += " " + sentence
    return textsummary


# In[19]:


# print(len(string_summary))
st.info(summarise_text(string_summary))


# In[20]:


# display the close price
st.header( " Close Price\n")
# st.line_chart(data_dated['4. close'])


# In[21]:



def plot_closingPrice(data_dated):
    closing_price= data_dated['4. close']
    coefficients, residuals, _, _, _ = np.polyfit(range(len(closing_price.index)),closing_price,1,full=True)
    mse = residuals[0]/(len(closing_price.index))
    nrmse = np.sqrt(mse)/(closing_price.max() - closing_price.min())


    color_override="green"
    if coefficients[0] < 0:
        color_override="red"


    fig = px.scatter(x=data_dated.index, y=data_dated['4. close'], trendline="ols", trendline_color_override=color_override)
    fig.update_layout(

        xaxis_title="Year",
        yaxis_title="Price",

        )

    # fig = px.line(trendline="ols")
    # fig.add_scatter(x=data_dated.index, y=data_dated['4. close'], name="Close")
    # fig= px.line([coefficients[0]*x + coefficients[1] for x in range(len(closing_price))])
    # fig.show()
    st.write(fig)


# In[22]:


    # interpreting the slope of the closing price
    if coefficients[0] > 0:
        st.info('Between {} to {} there has been an upward trend in the closing price of {} stock'.format(start_date, end_date, string_name))
    else:
        st.info('Between {} to {} there has been a downward trend in the closing price of {} stock'.format(start_date, end_date, string_name))

plot_closingPrice(data_dated)
# In[23]:


def show_corr():
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    # Draw the heatmap
    sns.heatmap(data_dated.corr(), center=0, cmap='Blues', annot=True)

    ax.set_title('Multi-Collinearity of Stock Attributes')
    st.write(fig)


# In[24]:

# display heatmap on UI
st.header(" Data Correlation")
show_corr()
st.info("On both axes, each feature (variable) is listed, and their relationships with other variables are coloured. The darker the colour, the more highly correlated those variables are and should not be paired in the same model. For this project, only variables that are highly correlated with Closing Price are considered. ")

# In[25]:


# get statistics on the data
st.header(" Descriptive Statistics")
st.write(data_dated.describe())


# In[26]:


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


# In[27]:


# calculate the change in price from the previous day
data_dated['change_in_price'] = data_dated['4. close'].diff()


# In[28]:


# *******************************************************Models****************************************************************
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


# In[29]:


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


# In[30]:


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


# In[31]:


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


# In[32]:


def calc_price_rate_of_change(data_dated):
    # Calculate the Price Rate of Change
    n = 9

    # Calculate the Rate of Change in the Price, and store it in the Data Frame.
    data_dated['Price_Rate_Of_Change'] = data_dated.groupby('Ticker')['4. close'].transform(
        lambda x: x.pct_change(periods=n))


# In[33]:


pd.options.mode.chained_assignment = None  # default='warn'

# Evaluation metrics
RMSE = []
Rsquared = []
adj_Rsquared= []
Mae = []


# In[34]:


# Building ridge regression model
def pricePrediction_LR(symbol, days, starting_date, end_date):

    #     obtain stock data
    stock_df = stock_data(symbol, starting_date, end_date)

    #     obtaining technical indicators
    stochastic_Oscillator(stock_df)
    calc_williams_r(stock_df)
    calc_macd(stock_df)
    calc_price_rate_of_change(stock_df)

    #     set the trading window we are trying to predict
    stock_df_targeted = trading_window(stock_df, days)

    stock_df_targeted.reset_index(inplace=True)
    stock_df_targeted = stock_df_targeted.dropna()

    stock_df_targeted_scaled = stock_df_targeted.copy()
    # drop unused columns in dataset
    stock_df_targeted_scaled.drop(
        ['Ticker', '4. close', '7. dividend amount', '3. low', '5. adjusted close', '6. volume', '8. split coefficient',
         'low_14', 'high_14', 'MACD_EMA'], axis=1, inplace=True)

    # scale feature and target
    target_price = stock_df_targeted_scaled.filter(['Target'])
    stock_df_targeted_scaled = sc.fit_transform(stock_df_targeted_scaled.drop(columns=['date','Target']))
    target_price = y_sc.fit_transform(target_price)

    # Creating Feature and Target
    X = stock_df_targeted_scaled[:, :6]
    y = target_price

    # splitting data into training and testing
    split = int(0.65 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]


# building and evaluating the model
    regression_model = Ridge(alpha=1)
    regression_model.fit(X_train, y_train)
    last_element = X_test[len(X_test) - 1]
    original_prices= stock_df_targeted['4. close'].values
    current_price= original_prices[len(original_prices) - 1]
    last_element=last_element.reshape(1, -1)


    # using model to predict the entire dataset
    predicted_prices = regression_model.predict(X)

    # using the model to predict the test data
    eval_predict = regression_model.predict(X_test)

    # predicting future price
    predicted_price= regression_model.predict(last_element)

    # calculating profits
    profitInXDays= y_sc.inverse_transform(predicted_price)
    share_amount= investment_amount/current_price
    profit=round((share_amount * profitInXDays[0][0])-investment_amount,2)

    # getting predicted prices for for the test data
    Predicted = []
    for i in predicted_prices:
        Predicted.append(i[0])
    # getting the original prices
    close = []
    for i in stock_df_targeted_scaled:
        close.append(i[0])
    # adding predicted and actual prices to a data frame
    df_predicted = stock_df_targeted[['date']]
    df_predicted['Close'] = close
    df_predicted['Prediction'] = Predicted

    # evaluating the prediction metrics
    RMSE.append(math.sqrt(mean_squared_error(y_test, eval_predict)))
    Rsquared.append(r2_score(y_test, eval_predict))
    Mae.append(mean_absolute_error(y_test, eval_predict))
    adj_Rsquared.append(1 - (1-r2_score(y_test, eval_predict))*(len(y_test)-1)/(len(y_test)-X.shape[1]-1))

    # appending information to web page

    interactive_plot(df_predicted, "Original Vs. Prediction")
    st.info("in {} day(s) the price of this stock will be £{}".format(days, round(profitInXDays[0][0], 2)))
    st.info("You would make £{} in {} day(s)".format(profit, days))

    unscaled_y_test = y_sc.inverse_transform(y_test)
    y_test_predicted = y_sc.inverse_transform(eval_predict)
    unscaled_y_test = [item for sublist in unscaled_y_test for item in sublist]
    y_test_predicted=  [item for sublist in y_test_predicted for item in sublist]
    data = {'Close': unscaled_y_test,
            'Prediction': y_test_predicted}

    st.header("Trading Algorithm")
    # Create DataFrame
    df = pd.DataFrame(data)
    trade_algorithm(df)
    plot_trades(unscaled_y_test, y_test_predicted, sells, buys)
    compute_earnings(buys, sells)
    # print(y_test_predicted)






# In[35]:


# randomForest Model
def pricePrediction_RandomForest(symbol, days, start_date, end_date):

    #     obtain stock data
    stock_df = stock_data(symbol, start_date, end_date)

    #     obtaining technical indicators
    stochastic_Oscillator(stock_df)
    calc_williams_r(stock_df)
    calc_macd(stock_df)
    calc_price_rate_of_change(stock_df)

    #     set the trading window we are trying to predict
    stock_df_targeted = trading_window(stock_df, days)

    # drop any row with null values
    stock_df_targeted.reset_index(inplace=True)
    stock_df_targeted = stock_df_targeted.dropna()

    stock_df_targeted_scaled = stock_df_targeted.copy()

    # dropping columns that were not used in the data
    stock_df_targeted_scaled.drop(
        ['Ticker', '4. close', '7. dividend amount', '3. low', '5. adjusted close', '6. volume', '8. split coefficient',
         'low_14', 'high_14', 'MACD_EMA'], axis=1, inplace=True)

    # scale feature and target
    target_price = stock_df_targeted_scaled.filter(['Target'])
    stock_df_targeted_scaled = sc.fit_transform(stock_df_targeted_scaled.drop(columns=['date','Target']))
    target_price = y_sc.fit_transform(target_price)

    # Creating Feature and Target
    X = stock_df_targeted_scaled[:, :6]
    y = target_price

    # splitting data into training and testing
    split = int(0.65 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]


# Building and fitting the model
    rf = RandomForestRegressor(n_estimators=1000)
    rf.fit(X_train, y_train.ravel())


    last_element = X_test[len(X_test) - 1]
    original_prices = stock_df_targeted['4. close'].values
    current_price = original_prices[len(original_prices) - 1]
    last_element = last_element.reshape(1, -1)
    # predicting future price
    predicted_price = rf.predict(last_element)
    predicted_price = predicted_price.reshape(1, -1)

    # making model prediction on the whole dataset
    pred_rf = rf.predict(X)
    # making predictions on the test data
    eval_predict= rf.predict(X_test)


    # calculating profits
    profitInXDays = y_sc.inverse_transform(predicted_price)
    share_amount = investment_amount / current_price
    profit = round((share_amount * profitInXDays[0][0]) - investment_amount, 2)

    # getting predicted prices for for the test data
    Predicted = []
    for i in pred_rf:
        Predicted.append(i)
    # getting the original prices
    close = []
    for i in stock_df_targeted_scaled:
        close.append(i[0])
    # adding predicted and actual prices to a data frame
    df_predicted = stock_df_targeted[['date']]
    df_predicted['Close'] = close
    df_predicted['Prediction'] = Predicted


    # evaluating the prediction metrics
    RMSE.append(math.sqrt(mean_squared_error(y_test, eval_predict)))
    Rsquared.append(r2_score(y_test, eval_predict))
    Mae.append(mean_absolute_error(y_test, eval_predict))
    adj_Rsquared.append(1 - (1 - r2_score(y_test, eval_predict)) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1))

    # plottig actual vs predicted graph
    interactive_plot(df_predicted, "Original Vs. Prediction for ")

    # appending information to web page
    st.info("in {} day(s) the price of this stock will be £{}".format(days, round(profitInXDays[0][0], 2)))
    st.info("You would make £{} in {} day(s)".format(profit, days))

# trading algorithm
    unscaled_y_test = y_sc.inverse_transform(y_test)
    y_test_predicted = y_sc.inverse_transform(eval_predict.reshape(1, -1))

    unscaled_y_test = [item for sublist in unscaled_y_test for item in sublist]
    y_test_predicted= [item for sublist in y_test_predicted for item in sublist]

    data = {'Close': unscaled_y_test,
            'Prediction': y_test_predicted}

    st.header("Trading Algorithm")
    # Create DataFrame
    df = pd.DataFrame(data)
    trade_algorithm(df)
    plot_trades(unscaled_y_test, y_test_predicted, sells, buys)
    compute_earnings(buys, sells)


# In[36]:


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

    stock_df_targeted_ = stock_df_targeted.copy()
    stock_df_targeted.drop(
        ['Ticker', '4. close', '7. dividend amount', '3. low', '5. adjusted close', '6. volume', '8. split coefficient',
         'low_14', 'high_14', 'MACD_EMA'], axis=1, inplace=True)

    stock_df_targeted.dropna(inplace=True)
    training_data_X = stock_df_targeted.iloc[:, 1:7].values
    training_data_y = stock_df_targeted.iloc[:, 7:].values

    stock_df_targeted_scaled = sc.fit_transform(stock_df_targeted.drop(columns=['date']))

    X = sc.fit_transform(training_data_X)
    y = y_sc.fit_transform(training_data_y)

    # Convert the data into array format
    X = np.asarray(X)
    y = np.asarray(y)

    # Split the data
    split = int(0.65 * len(X))
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

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    predicted = model.predict(X)
    eval_predict = model.predict(X_test)

    last_element = X_test[len(X_test) - 1]
    original_prices = stock_df_targeted_['4. close'].values
    current_price = original_prices[len(original_prices) - 1]
    last_element = last_element.reshape(1, -1)

    # predicting future price
    predicted_price = model.predict(last_element)
    predicted_price = predicted_price.reshape(1, -1)

    # calculating profits
    profitInXDays = y_sc.inverse_transform(predicted_price)
    profitInXDays= round(profitInXDays[0][0], 2)
    share_amount = investment_amount / current_price
    profit = round((share_amount * profitInXDays) - investment_amount, 2)


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

# add evaluations to list
    RMSE.append(math.sqrt(mean_squared_error(y_test, eval_predict)))
    Rsquared.append(r2_score(y_test, eval_predict))
    Mae.append(mean_absolute_error(y_test, eval_predict))
    adj_Rsquared.append(1 - (1 - r2_score(y_test, eval_predict)) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1))

    # Plot the data
    interactive_plot(df_predicted, "Original Vs Prediction")
    # appending information to web page

    st.info("in {} day(s) the price of this stock will be £{}".format(days, round(profitInXDays, 2)))
    st.info("You would make £{} in {} day(s)".format(profit, days))


    unscaled_y_test = y_sc.inverse_transform(y_test)
    y_test_predicted = y_sc.inverse_transform(eval_predict)
    unscaled_y_test = [item for sublist in unscaled_y_test for item in sublist]
    y_test_predicted = [item for sublist in y_test_predicted for item in sublist]
    data = {'Close': unscaled_y_test,
            'Prediction': y_test_predicted}

    st.header("Trading Algorithm using Testing Data")
    # Create DataFrame
    df = pd.DataFrame(data)
    trade_algorithm(df)
    plot_trades(unscaled_y_test, y_test_predicted, sells, buys)
    compute_earnings(buys, sells)




# trade algorithm
def trade_algorithm(df_predicted):
    x=0
    for  actual , predicted  in zip( df_predicted['Close'], df_predicted['Prediction']):

        predicted_price=predicted
        price_today=actual

        delta= predicted_price- price_today

        if delta > thresh:
            buys.append((x,price_today))
        elif delta <= thresh:
            sells.append((x,price_today))
        x +=1

# plot trades from trading algorithm
def plot_trades(y_test_predicted, unscaled_y_test, sells, buys):
    fig= go.Figure()
    start = 0
    end = -1
    fig.add_trace(
        go.Scatter(
            x=list(list(zip(*buys))[0]),
            y=list(list(zip(*buys))[1]),
            marker=dict(color="green", size=6),
            mode="markers",
            name="Buy",

    ))

    fig.add_trace(
        go.Scatter(
            x=list(list(list(zip(*sells))[0])),
            y=list(list(list(zip(*sells))[1])),
            marker=dict(color="red", size=6),
            mode="markers",
            name="Sell",

        ))

    fig.add_trace(
        go.Scatter(
            y=unscaled_y_test[start:end],
            marker=dict(color="gold", size=6),
            name="Actual",

        ))

    fig.add_trace(
        go.Scatter(
            y=y_test_predicted[start:end],
            marker=dict(color="blue", size=6),
            name="Predicted",

        ))

    fig.update_layout(
        xaxis_title="Trade Number",
        yaxis_title="Price (£)",

    )
    st.write(fig)

# compute the earnings from the trading algorithm
def compute_earnings(buys, sells):
    purchase_amt = investment_amount
    stock = 0
    balance = 0
    while len(buys) > 0 and len(sells) > 0:
        if buys[0][0] < sells[0][0]:
            # time to buy $10 worth of stock
            balance -= purchase_amt
            stock += purchase_amt / buys[0][1]
            buys.pop(0)
        else:
            # time to sell all of our stock
            balance += stock * sells[0][1]
            stock = 0
            sells.pop(0)
    st.info("Profit made from test data is £{}".format(balance))






# check with model the user has picked
if model == 'Ridge Regression':
    st.header("Ridge Regression Model for " + str(num_days) + " Day(s)")
    # Pretend we're doing some computation that takes time.
    progress_bar = st.progress(0)

    for i in range(100):
        # Update progress bar.
        progress_bar.progress(i + 1)

    progress_bar.balloons()

    pricePrediction_LR(symbol, num_days, start_date, end_date)
elif model == 'Random Forest':
    st.header("Random Forest model for " + str(num_days) + " Day(s)")
    pricePrediction_RandomForest(symbol, num_days, start_date, end_date)
elif model == 'LSTM':

    st.header("LSTM Model for " + str(num_days) + " Day(s)")
    pricePrediction_LSTM(symbol, num_days, start_date, end_date)


# In[37]:


st.header("Evaluation metrics")
# add evaluation metrics to dataframe
evaluation_metrics = {'Rsquared': Rsquared,  'Adj_Rsquared': adj_Rsquared,'MAE': Mae, 'RMSE': RMSE}

# add eval metrics to table
chart = st.table(evaluation_metrics)




