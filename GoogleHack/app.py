import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from datetime import date
from datetime import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from yahoo_fin import stock_info as si
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
import matplotlib
from wordcloud import WordCloud
from textblob import TextBlob
import GetOldTweets3 as got
import requests
import ssl
import urllib
import urllib.parse
import urllib.error
from difflib import SequenceMatcher
from datetime import timedelta
import requests
from bs4 import BeautifulSoup
import ssl
import urllib.parse
import urllib.error
matplotlib.use('Agg')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
ssl._create_default_https_context = ssl._create_unverified_context

from urllib.request import Request, urlopen

import csv

st.title('Business Insights')


def livepass(passw):
    if passw == 'Google' or passw == 'Pagerank':
        st.subheader('Financial Analysis')
        st.write('Welcome to our professional business dashboard, here you can find information '
                 'about finance, overall reputation and social contribution of any company you like.')

        company = st.text_input('Company')
        stock_type = st.text_input('First Stock Parameter')
        stock_type2 = st.text_input('Second Stock Parameter')
        start_date = st.text_input('Start Date')
        end_date = st.text_input('End Date')

        if company and stock_type and stock_type2 and start_date and end_date:
            st.write("Thanks for adding input source")
        else:
            company = "GOOGL"
            stock_type = "Close"
            stock_type2 = "Open"
            start_date = "2017-08-05"
            end_date = "2018-10-20"

        latest = []
        time = []


        def graph_new(latest, time, company):
            for i in range(0, 5):
                latest.append(si.get_live_price(company))
                time.append(datetime.datetime.now())
            return latest, time


        if end_date == str(datetime.datetime.now().date()):
            latest, time = graph_new(latest, time, company)

        d = pd.Series(latest, index=time)
        df = yf.download(company, start_date, end_date)

        trace1 = []
        trace2 = []
        trace1.append(go.Scatter(x=df[stock_type].append(d).index, y=df[stock_type].append(d), mode='lines',
            opacity=0.7,name= company + ' ' + stock_type,textposition='bottom center'))
        trace2.append(go.Scatter(x=df[stock_type2].append(d).index,y=df[stock_type2].append(d),mode='lines',
            opacity=0.6,name=company + ' ' + stock_type2,textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"Company Stocks",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}

        st.plotly_chart(figure)

        st.write('')
        st.write('')
        st.subheader('Competitor Analysis')
        corr = st.text_input('Competitor')
        option = st.selectbox('Which stock parameter do you want to analyse?', ('High', 'Low', 'Open', 'Close', 'Volume'))
        trace1 = []
        if corr != '':
            df2 = yf.download(corr, start_date, end_date)
            trace1.append(
                go.Scatter(x=df[option], y=df2[option],
                           mode='markers', opacity=0.7, textposition='bottom center'))
            traces = [trace1]
            data = [val for sublist in traces for val in sublist]
            figure = {'data': data,
                      'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                          height=600, title=f"{option} of {company} vs {corr} Over Time",
                                          xaxis={"title": company, }, yaxis={"title": corr}, paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)')}
            st.plotly_chart(figure)


        def create_dataset(dataset, window_size=1):
            data_X, data_Y = [], []
            for i in range(len(dataset) - window_size - 1):
              a = dataset[i:(i + window_size), 0]
              data_X.append(a)
              data_Y.append(dataset[i + window_size, 0])
            return (np.array(data_X), np.array(data_Y))


        def fit_model(train_X, train_Y, window_size=1):
            model = Sequential()

            model.add(LSTM(4,
                         input_shape=(1, window_size)))
            model.add(Dense(1))
            model.compile(loss="mean_squared_error",
                        optimizer="adam")
            model.fit(train_X,
                    train_Y,
                    epochs=3,
                    batch_size=1,
                    verbose=2)
            return model


        def predict_and_score(model, X, Y, scaler):
            # Make predictions on the original scale of the data.
            pred = scaler.inverse_transform(model.predict(X))
            # Prepare Y data to also be on the original scale for interpretability.
            orig_data = scaler.inverse_transform([Y])
            # Calculate RMSE.
            score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
            return (score, pred)


        def LSTMPreds(df, name):
            data = df[name].values.astype("float32")

            # Applying the MinMax scaler from sklearn
            # to normalize data in the (0, 1) interval.
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(data.reshape(-1, 1))

            # Using 75% of data for training, 25% for validation.
            TRAIN_SIZE = 0.75

            train_size = int(len(dataset) * TRAIN_SIZE)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
            #print("Number of entries (training set, test set): " + str((len(train), len(test))))

            # Create test and training sets for one-step-ahead regression.
            window_size = 12
            train_X, train_Y = create_dataset(train, window_size)
            test_X, test_Y = create_dataset(test, window_size)
            #print("Original training data shape:")
            #print(train_X.shape)

            # Reshape the input data into appropriate form for Keras.
            train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
            test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
            #print("New training data shape:")
            #print(train_X.shape)

            # Create test and training sets for one-step-ahead regression.
            window_size = 12
            train_X, train_Y = create_dataset(train, window_size)
            test_X, test_Y = create_dataset(test, window_size)
            #print("Original training data shape:")
            #print(train_X.shape)

            # Reshape the input data into appropriate form for Keras.
            train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
            test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
            #print("New training data shape:")
            #print(train_X.shape)

            model1 = fit_model(train_X, train_Y, window_size)

            rmse_train, train_predict = predict_and_score(model1, train_X, train_Y, scaler)
            rmse_test, test_predict = predict_and_score(model1, test_X, test_Y, scaler)

            t = scaler.inverse_transform([test_Y])

            st.write("Training data score: %.2f RMSE" % rmse_train)
            st.write("Test data score: %.2f RMSE" % rmse_test)

            # Start with training predictions.
            train_predict_plot = np.empty_like(dataset)
            train_predict_plot[:, :] = np.nan
            train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict

            # Add test predictions.
            test_predict_plot = np.empty_like(dataset)
            test_predict_plot[:, :] = np.nan
            test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict

            # Create the plot.
            fig, ax = plt.subplots(figsize=(18, 10))
            ax.plot(df.index, scaler.inverse_transform(dataset), label = "True value")
            ax.plot(df.index, train_predict_plot, label = "Training set prediction")
            ax.plot(df.index, test_predict_plot, label = "Test set prediction")
            plt.xlabel("Time")
            plt.ylabel("Close Price")
            plt.title("Comparison true vs. predicted training / test")
            plt.legend()
            st.pyplot()

            return test_predict, t


        def predreport(y_pred, Y_Test):
            diff = y_pred.flatten() - Y_Test.flatten()
            perc = (abs(diff) / y_pred.flatten()) * 100
            priority = []
            for i in perc:
              if i > 2:
                  priority.append(3)
              elif i > 1:
                  priority.append(2)
              else:
                  priority.append(1)

            st.write("Error Importance 1 reported in ", priority.count(1), "cases \n")
            st.write("Error Importance 2 reported in ", priority.count(2), "cases \n")
            st.write("Error Importance 3 reported in ", priority.count(3), "cases \n")
            colors = ['rgb(102, 153, 255)', 'rgb(0, 255, 0)', 'rgb(255, 153, 51)',
                    'rgb(255, 51, 0)']

            fig = go.Figure(
              data=[go.Table(header=dict(values=['Actual Values', 'Predictions', '% Difference', "Error Importance"],
                                         line_color=[np.array(colors)[0]],
                                         fill_color=[np.array(colors)[0]],
                                         align='left'),
                             cells=dict(values=[y_pred.flatten(), Y_Test.flatten(), perc, priority],
                                        line_color=[np.array(colors)[priority]], fill_color=[np.array(colors)[priority]],
                                        align='left'))
                    ])
            st.plotly_chart(fig)


        st.write('')
        st.write('')
        st.subheader('Long Short Term Memory Analysis')
        y_pred , Y_Test = LSTMPreds(df, stock_type)
        st.subheader('Prediction Summary Table')
        predreport(y_pred, Y_Test)

        ssl._create_default_https_context = ssl._create_unverified_context

        ### Overall

        def get_metrics(company_name, start_date, end_date, twitter_username, instagram_username,
                        search_terms,
                        max_tweets=1000):
            tweets_about = get_tweets_about_company(company_name, search_terms, start_date, end_date, max_tweets)
            twitter_sentiment = get_twitter_sentiment(tweets_about)
            user_engagement = get_user_engagement(twitter_username, start_date, end_date, max_tweets)
            instagram_followers = get_insta_info(instagram_username)
            twitter_followers = get_twitter_followers(twitter_username)
            social_media_reach = max(instagram_followers, twitter_followers)
            upload_word_cloud(company_name, tweets_about, descriptions=[])

            return {
                "twitter_sentiment": twitter_sentiment,
                "user_engagement": user_engagement,
                "reach": social_media_reach,
                "over_time_plot": sentiment_over_time(company_name, search_terms, start_date, end_date, max_tweets)
            }

        def upload_word_cloud(company_name, tweets, descriptions, max_words=50):
            text_list = []
            for tweet in tweets:
                tweet_blob = TextBlob(tweet.text.lower())
                for word in tweet_blob.words:
                    if str(word) != company_name.lower() and str(word) not in "https":
                        text_list.append(word)

            for description in descriptions:
                text_blob = TextBlob(description.lower())
                for word in text_blob.words:
                    if str(word) != company_name.lower() and str(word) not in "https":
                        text_list.append(word)

            text = " ".join(text_list)
            word_cloud = WordCloud(max_words=max_words).generate(text)
            plt.imshow(word_cloud, interpolation="bilinear")
            plt.axis("off")
            plt.legend('')
            st.pyplot()

        def sentiment_over_time(company_name, search_terms, start_date,
                                end_date, max_tweets=1000):
            times, tweet_list = get_tweets_about_company_over_time(company_name, search_terms, start_date, end_date,
                                                                   max_tweets)
            sentiments = []
            for i in range(len(times)):
                sentiments.append(get_twitter_sentiment(tweet_list[i]))
            return times, sentiments

        ### Instagram

        def get_insta_info(user_name):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            url = "https://www.instagram.com/" + user_name
            html = urllib.request.urlopen(url, context=ctx).read()
            soup = BeautifulSoup(html, 'html.parser')
            data = soup.find_all('meta', attrs={'property': 'og:description'})
            text = data[0].get('content').split()
            followers = text[0]
            followers = followers.replace('k', '*1e3')
            followers = followers.replace('m', '*1e6')
            followers_num = eval(followers)
            return followers_num

        ### Twitter

        def get_twitter_followers(company_username):
            url = "https://cdn.syndication.twimg.com/widgets/followbutton/info.json?screen_names=" + company_username
            response = requests.get(url)
            return int(response.json()[0]['followers_count'])

        def get_tweets_from_company(company_username, start_date,
                                    end_date,
                                    max_tweets=1000):
            tweet_criteria = got.manager.TweetCriteria().setUsername(company_username) \
                .setSince(start_date.strftime("%Y-%m-%d")) \
                .setUntil(end_date.strftime("%Y-%m-%d")) \
                .setMaxTweets(max_tweets)
            tweets = got.manager.TweetManager.getTweets(tweet_criteria)
            return tweets

        def get_user_engagement(company_username, start_date,
                                end_date,
                                max_tweets=1000):
            tweets = get_tweets_from_company(company_username, start_date, end_date, max_tweets)
            engagements = 0
            for tweet in tweets:
                engagements += tweet.favorites + tweet.retweets
            followers = get_twitter_followers(company_username)
            return engagements * len(tweets) / followers

        def get_tweets_about_company_over_time(company_name, search_terms, start_date,
                                               end_date,
                                               max_tweets=1000):
            times = []
            tweet_lists = []
            days = (end_date - start_date).days + 1
            tweets_per_day = max_tweets / days + 1
            for day in range(days):
                current_date = start_date + timedelta(days=day)
                times.append(current_date)
                tweet_lists.append(
                    get_tweets_about_company(company_name, search_terms, current_date, current_date + timedelta(days=1),
                                             tweets_per_day))
            return times, tweet_lists

        def get_tweets_about_company(company_name, search_terms, start_date,
                                     end_date,
                                     max_tweets=1000):
            tweet_criteria = got.manager.TweetCriteria().setQuerySearch(company_name + " " + search_terms) \
                .setSince(start_date.strftime("%Y-%m-%d")) \
                .setUntil(end_date.strftime("%Y-%m-%d")) \
                .setMaxTweets(max_tweets)
            tweets = got.manager.TweetManager.getTweets(tweet_criteria)
            return tweets

        def get_twitter_sentiment(tweets, retweets_weight=0.9, likes_weight=0.5):
            total_amount = 0
            total_sentiment = 0
            for tweet in tweets:
                tweet_blob = TextBlob(tweet.text)
                amount = 1 + tweet.favorites * likes_weight + tweet.retweets * retweets_weight
                total_sentiment += tweet_blob.sentiment.polarity * amount * (
                            2 - (tweet_blob.sentiment.subjectivity + 1) / 2.0)
                total_amount += amount
            if total_amount == 0:
                return 0
            return total_sentiment / total_amount

        def get_symbol(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

            result = requests.get(url).json()

            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']

        def get_green_rankings(ticker):

            user_input = ticker

            company = get_symbol(user_input)

            results = []

            with open("environment.csv", encoding='utf-8-sig') as csvfile:
                reader = csv.reader(csvfile)  # change contents to floats
                for row in reader:  # each row is a list
                    results.append(row)

            ranking = -1

            print(company)
            print(SequenceMatcher(None, company, 'Alphabet Inc').ratio())

            for x in results:
                if (SequenceMatcher(None, company, x[1]).ratio()) > 0.8:
                    ranking = x[0]

            return ranking

        def get_engagement(ticker):

            url = ('https://engagements.ceres.org/')
            dfs = pd.read_html(url)
            df = dfs[0]

            user_input = ticker
            company = get_symbol(user_input)

            engagements = []

            for x in df.itertuples():
                if (SequenceMatcher(None, company, x[3]).ratio()) > 0.99:
                    engagements.append([x[1], x[2]])

            return engagements

        st.subheader('Coorperate Social Responsability (CSR)')
        ranking = get_green_rankings(company)
        st.write(company + " position in the top 500 most valuables industry: ", ranking)
        eng = get_engagement(company)
        st.write("Social Engagement")
        st.write(pd.DataFrame({'Status': [eng[0][0], eng[1][0], eng[2][0], eng[3][0]],
                               'Sustainability Proposal': [eng[0][1], eng[1][1], eng[2][1], eng[3][1]],}))

        st.subheader('Sentiment Analysis')
        name = st.text_input('Name')
        start = st.text_input('Start Date')
        end = st.text_input('End Date')
        twitter_username = st.text_input('Twitter Username')
        instagram_username = st.text_input('Instagram Username')
        search_terms = st.text_input('Search Terms')
        if name and start and end and twitter_username and instagram_username and search_terms:
            metrics = get_metrics(name, datetime.datetime.strptime(start, "%Y-%m-%d"),
                        datetime.datetime.strptime(end, "%Y-%m-%d"), twitter_username,
                        instagram_username, search_terms)
            st.write("Twitter Sentiment", metrics['twitter_sentiment'])
            st.write("User Engagement", metrics['user_engagement'])
            st.write("Reach", metrics['reach'])
            plt.axis("on")
            x, y = metrics['over_time_plot']

            def convert(s):
                return datetime.datetime.strptime(s, '%Y-%m-%d')

            someobject = convert(start)
            someobject2 = convert(end)
            xn = [convert(start)]
            for i in range(0, abs((someobject - someobject2).days)):
                xn.append(convert(start) + timedelta(days=i+1))

            res = []
            for i in xn:
                res.append(i.strftime('%Y-%m-%d'))
            st.write("Last 3 days company trends: ")
            st.write(pd.DataFrame({'Date': [res[0], res[1], res[2]],
                                   'Sentiment Score': [y[0], y[1], y[2]], }))

    else:
        st.write("Access Denied")


passw = st.text_input('Please enter your password: ')
livepass(passw)


