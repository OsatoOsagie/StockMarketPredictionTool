#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install yahoo-finance


# In[2]:


# pip install alpha_vantage pandas


# In[3]:


# pip install yfinance


# In[46]:


pip install statsmodels


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
import time
import yfinance as yf
# import statsmodels.api as sm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')


# In[6]:


# App title and Credits
st.markdown('''
# Stock Price App
Shown are the stock price data for query companies!
**Credits**
- App built by [Osato Osagie](https://www.linkedin.com/in/osato-osagie) (aka [HenchTechGuy](https://github.com/greggs25))
- Built in `Python` using `Streamlit`,`yfinance`,`AlphaVantage`, `Scikit-learn`, `Tensorflow`,`Pandas`, `Plotly`, `Math` and `Seaborn`
''')
st.write('---')


# In[7]:


df_tickers = pd.read_csv('nasdaq_screener.csv')


# In[8]:


sc = MinMaxScaler(feature_range=(0, 1))


# In[9]:


st.sidebar.header('Query parameters')

# present date
today = date.today()


# In[10]:


# function to extract the list of functions
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
    num_days = st.sidebar.selectbox(
        'Select Number of Days',
        (1, 2, 3, 4, 5, 6, 7))
    model = st.sidebar.selectbox(
        'Select Model',
        (['Ridge Regression', 'Random Forest', 'LSTM']))
    return start_date, end_date, stock_symbol, num_days, model


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
    st.write(fig)


# In[14]:


# function for plotting the closing price
def interactive_plot_closingPrice(df, title):
    fig = px.line()
    fig.add_scatter(x=df.index, y=df['4. close'], name="Close")
    st.write(fig)


# In[15]:


# get users input
start_date, end_date, symbol, num_days, model = get_input()


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
    # Input text - to summarize
    text = string_summary

    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Creating a frequency table to keep the
    # score of each word

    frequencyTable  = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in frequencyTable:
            frequencyTable[word] += 1
        else:
            frequencyTable[word] = 1

    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in frequencyTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumOfValues  = 0
    for sentence in sentenceValue:
        sumOfValues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumOfValues / len(sentenceValue))

    # Storing sentences into our summary.
    textsummary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
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




closing_price= data_dated['4. close']
coefficients, residuals, _, _, _ = np.polyfit(range(len(closing_price.index)),closing_price,1,full=True)
mse = residuals[0]/(len(closing_price.index))
nrmse = np.sqrt(mse)/(closing_price.max() - closing_price.min())
print('Slope ' + str(coefficients[0]))
print('NRMSE: ' + str(nrmse))

color_override="green"
if coefficients[0] < 0:
    color_override="red"


fig = px.scatter(x=data_dated.index, y=data_dated['4. close'], trendline="ols", trendline_color_override=color_override)
fig.show()

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


# In[23]:


def show_corr():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data_dated.corr(), center=0, cmap='Blues', annot=True)
    ax.set_title('Multi-Collinearity of Car Attributes')
    st.write(fig)


# In[24]:


st.header(" Data Correlation")
show_corr()


# In[25]:


# get statistics on the data
st.header(" Data Statistics")
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
    adj_Rsquared.append(1 - (1-r2_score(y, predicted_prices))*(len(y)-1)/(len(y)-X.shape[1]-1))

    interactive_plot(df_predicted, "Original Vs. Prediction")


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
    adj_Rsquared.append(1 - (1 - r2_score(y, pred_rf)) * (len(y) - 1) / (len(y) - X.shape[1] - 1))
    interactive_plot(df_predicted, "Original Vs. Prediction for ")


# In[ ]:





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
    stock_df_targeted.drop(
        ['Ticker', '4. close', '7. dividend amount', '3. low', '5. adjusted close', '6. volume', '8. split coefficient',
         'low_14', 'high_14', 'MACD_EMA'], axis=1, inplace=True)
    stock_df_targeted.dropna(inplace=True)
    training_data_X = stock_df_targeted.iloc[:, 1:7].values
    training_data_y = stock_df_targeted.iloc[:, 7:].values

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

# add evaluations to list
    RMSE.append(math.sqrt(mean_squared_error(y, predicted)))
    Rsquared.append(r2_score(y, predicted))
    Mae.append(mean_absolute_error(y, predicted))
    adj_Rsquared.append(1 - (1 - r2_score(y, predicted)) * (len(y) - 1) / (len(y) - X.shape[1] - 1))

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
    progress_bar.balloons()
elif model == 'Random Forest':
    st.header("Random Forest model for " + str(num_days) + " Day(s)")
    pricePrediction_RandomForest(symbol, num_days, start_date, end_date)
elif model == 'LSTM':

    st.header("LSTM Model for " + str(num_days) + " Day(s)")
    pricePrediction_LSTM(symbol, num_days, start_date, end_date)


# In[37]:


st.header("Evaluation metrics")
# add evaluation metrics to dataframe
evaluation_metrics = {'Rsquared': Rsquared,  'Adj_Rsquared': adj_Rsquared,'Mae': Mae, 'RMSE': RMSE}

# add eval metrics to table
chart = st.table(evaluation_metrics)


# In[38]:


# calculate the number of days that have elapsed in chosen time 
time_elapsed = (data_dated.index[-1] - data_dated.index[0]).days


# In[39]:


#Current price / first record (e.g. price at beginning of 2009)

#provides us with the total growth %

total_growth = (data_dated['5. adjusted close'][-1] / data_dated['5. adjusted close'][1])

#Next, I want to annualize this percentage

#First, I convert our time elapsed to the # of years elapsed

number_of_years = time_elapsed / 365.0
#Second, I can raise the total growth to the inverse of the # of years

#(e.g. ~1/10 at time of writing) to annualize our growth rate

cagr = total_growth ** (1/number_of_years) - 1


#Now that we have the mean annual growth rate above,

#we'll also need to calculate the standard deviation of the

#daily price changes

std_dev = data_dated['5. adjusted close'].pct_change().std()



#Next, because there are roughy ~252 trading days in a year,

#we'll need to scale this by an annualization factor

#reference: https://www.fool.com/knowledge-center/how-to-calculate-annualized-volatility.aspx



number_of_trading_days = 252

std_dev = std_dev * math.sqrt(number_of_trading_days)



#From here, we have our two inputs needed to generate random

#values in our simulation

print ("cagr (mean returns) : ", str(round(cagr,4)))

print ("std_dev (standard deviation of return : )", str(round(std_dev,4)))


# In[40]:


#Generate random values for 1 year's worth of trading (252 days),

#using numpy and assuming a normal distribution

daily_return_percentages = np.random.normal(cagr/number_of_trading_days, std_dev/math.sqrt(number_of_trading_days),number_of_trading_days)+1



#Now that we have created a random series of future

#daily return %s, we can simply apply these forward-looking

#to our last stock price in the window, effectively carrying forward

#a price prediction for the next year



#This distribution is known as a 'random walk'



price_series = [data_dated['5. adjusted close'][-1]]



for j in daily_return_percentages:

    price_series.append(price_series[-1] * j)



#Great, now we can plot of single 'random walk' of stock prices

plt.plot(price_series)

plt.show()


# In[41]:


#Now that we've created a single random walk above,

#we can simulate this process over a large sample size to

#get a better sense of the true expected distribution

number_of_trials = 3000



#set up an additional array to collect all possible

#closing prices in last day of window.

#We can toss this into a histogram

#to get a clearer sense of possible outcomes

closing_prices = []



for i in range(number_of_trials):

    #calculate randomized return percentages following our normal distribution

    #and using the mean / std dev we calculated above

    daily_return_percentages = np.random.normal(cagr/number_of_trading_days, std_dev/math.sqrt(number_of_trading_days),

number_of_trading_days)+1

    price_series = [data_dated['5. adjusted close'][-1]]



    for j in daily_return_percentages:

        #extrapolate price out for next year

        price_series.append(price_series[-1] * j)



    #append closing prices in last day of window for histogram

    closing_prices.append(price_series[-1])



    #plot all random walks

    plt.plot(price_series)







plt.show()



#plot histogram



plt.hist(closing_prices,bins=40)



plt.show()


# In[42]:


mean_end_price = round(np.mean(closing_prices),2)

print("Expected price: ", str(mean_end_price))


# In[43]:


#lastly, we can split the distribution into percentiles

#to help us gauge risk vs. reward



#Pull top 10% of possible outcomes

top_ten = np.percentile(closing_prices,100-10)



#Pull bottom 10% of possible outcomes

bottom_ten = np.percentile(closing_prices,10);

fig, ax = plt.subplots()

#create histogram again

ax.hist(closing_prices,bins=40)

#append w/ top 10% line

ax.axvline(top_ten,color='r',linestyle='dashed',linewidth=2)

#append w/ bottom 10% line

ax.axvline(bottom_ten,color='r',linestyle='dashed',linewidth=2)

#append with current price

ax.axvline(data_dated['5. adjusted close'][-1],color='g', linestyle='dashed',linewidth=2)



# plt.show()


# In[44]:


st.header("Monte Carlo Simulation")
st.pyplot(fig)
# st.write(mean_end_price)


# In[45]:


if mean_end_price >data_dated['5. adjusted close'][-1]:
    st.info('Based solely on Monte Carlo simulations, the next year appears to have more upside than downside for {}, it has an expected running price of ${} and only 10 percent chance of the price landing below ${}.'.format(string_name, mean_end_price, round(np.percentile(closing_prices,10),2)))
else:
    st.info('Based solely on Monte Carlo simulations, the next year appears to have more downside than upside for {}, it has an expected running price of ${}.'.format(string_name, mean_end_price))


# In[ ]:





# In[ ]:




