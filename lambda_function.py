## at someone. auto reply a tweet


from doctest import OutputChecker
from logging import root
from re import S
import tweepy
import os
import random
import json
import csv
import pickle
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import regex
import boto3


from pathlib import Path
from pprint import pprint
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

pd.options.mode.chained_assignment = None  

from nltk.tokenize import punkt
nltk.data.path.append("/var/task/nltk_data")


######################################################################
### Variables
# Gloabl variables
ROOT = os.getcwd()
date = datetime.today().strftime('%Y%m%d')
month = datetime.today().strftime('%Y%m')
time = datetime.now().strftime('%H%M')

# Environment variables
query = os.getenv("QUERY")
tweets_cnt = int(os.getenv("TWEETS_CNT"))

bearer_token = os.getenv("BEARER_TOKEN")
consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


client = tweepy.Client(bearer_token=bearer_token,
                    consumer_key=consumer_key,
                    consumer_secret=consumer_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret)


######################################################################
### Main function

def lambda_handler(event, context):  

    # Action_01: Authenticate to Twitter

    # Action_02: Collect Twitter raw data
    tweets_info = searchTweets(client, query, tweets_cnt)

    # Action_03: Clean Tweets data
    tweets_text_orig, tweets_text_tokenize, tweets_text_split = Tweets_Clean_Data(tweets_info)

    # Action_04_A: Top10 popular words(without stopwords)
    tweets_top10_words = Tweets_Analysis_pplwds_top10(tweets_text_tokenize)
    intro = 'Daily top10 popular words of #Bitcoin on twitter (' + date + '):\n'    
    end = '---------------------\nby @BTCdailyData'
    text = str(intro) + tweets_top10_words
    Tweets_Create(text)

    # Action_04_B: Top10 popular hashtag
    hashtag_top10 = Tweets_Analysis_hashtag_top10(tweets_text_split)
    intro = 'Daily top10 hashtag of #Bitcoin on twitter (' + date + '): \n'    
    end = '---------------------\nby @BTCdailyData'
    text = str(intro) + hashtag_top10
    Tweets_Create(text)

    # Action_04_C: Top10 mentioned username
    usernames_mentioned_top10 = Tweets_Analysis_mtdusr_top10(tweets_text_split)

    # Action_04_D: most common sources
    """
    source_top10 = Tweets_Analysis_sources_top10(tweets_info)
    intro = 'Daily top10 source of tweets about #Bitcoin (' + date + '): \n'    
    end = '---------------------\nby @BTCdailyData'
    text = str(intro) + source_top10
    Tweets_Create(text)
    """

    # Action_04_E: most influential tweets
    tweets_influ_top10 = Tweets_Analysis_influential_tweets_top10(tweets_info)
    for i in range(0,3):
        id = tweets_influ_top10.loc[i]['tweet_id']
        text =  'Daily top' + str(i+1) + ' popular tweet on the topic of #Bitcoin (' + date + '):'
        client.create_tweet(text=text, quote_tweet_id=id)


    # Action_05: word cloud of tweets
    Word_Cloud(tweets_text_tokenize)

    # Action_06_A: Average of polarity and subjectivity
    tweets_pol_mean, tweets_sub_mean = Tweets_Analysis_sentiment_avg(tweets_info)

    # Action_06_B: Most happy/sadness tweets
    df_tweets_senti = Tweets_Sentiment_Clean_Data(tweets_info)
    df_tweets_ord_top10, df_tweets_ord_buttom10 = Tweets_Analysis_sentiment_top10(df_tweets_senti)

    # Action_07 create tweets
#    Tweets_Create(source_top10)


######################################################################
### functions    

## collect tweets info
def searchTweets(client, query, tweets_cnt):

    tweet_fields=['author_id', 'created_at', 'lang', 'possibly_sensitive', 'source', 'geo', 'entities', 'public_metrics', 'context_annotations']
    user_fields=['id', 'name', 'username', 'created_at','profile_image_url','public_metrics']
    expansions = ['author_id', 'referenced_tweets.id', 'geo.place_id', 'attachments.media_keys', 'in_reply_to_user_id']
    start_time = None; 
    end_time = None
    #start_time = '2022-11-04T09:00:00Z'
    #end_time = '2022-10-24T10:00:00Z'

    tweets = tweepy.Paginator(client.search_recent_tweets, 
                    query=query, 
                    tweet_fields=tweet_fields, 
                    user_fields=user_fields,
                    expansions=expansions,
                    start_time=start_time,
                    end_time=end_time,
                    max_results=100).flatten(limit=tweets_cnt)
   
    tweets_info = []

    for tweet in tweets: 
        tweets_info.append(tweet.data)

    # from list to dict 
    # the following code of "result" is cited from TwitterCollector by Gene Moo Lee, Jaecheol Park and Xiaoke Zhang
    result = {}
    result['collection_type'] = 'recent post'
    result['query'] = query
    result['tweet_cnt'] = len(tweets_info)
    result['tweets'] = tweets_info

    # save result to file
    file_name = date + '_bitcoin_' + str(tweets_cnt) + '.json'
    Save_Result_to_File(result,file_name)

    # save file to s3
    #local_file = '/tmp/Data/' + file_name
    #s3_file = os.path.join('Data_Tweets/', month, file_name)
    #Write_to_s3(local_file, s3_file)

    return result


## clean tweets data (without stopwords)
def Tweets_Clean_Data(tweets_info):

    tweets_text_orig = []

    for i in range(0, len(tweets_info['tweets'])):
        tweets_text_orig.append(tweets_info['tweets'][i]['text'])

    # tweets_text_1: tweets split 
    tweets_text_1 = str(tweets_text_orig).lower()
    tweets_text_t = nltk.word_tokenize(tweets_text_1)
    tweets_text_split = tweets_text_1.split()

    os.chdir(ROOT)

    with open('stopwords.pkl', 'rb') as file:
        stopwords = pickle.load(file)
    
    stopwords_add = ['https', 'amp','amp;','`','#','i’ve','she’s','would','ha']

    stopwords += stopwords_add

    stopsymbol = ["’","/",".","#",'``']

    tweets_text_tokenize = []

    for w in tweets_text_t:
        if w not in stopwords and len(w)>1:
            if '.' not in w and "'" not in w and "\\" not in w and '``' not in w:
                tweets_text_tokenize.append(w)
    
    return tweets_text_orig, tweets_text_tokenize, tweets_text_split



## 10 most popular words without stop words
def Tweets_Analysis_pplwds_top10(tweets_text_tokenize):
    
    tweets_word_count = Counter(tweets_text_tokenize)
    tweets_word_count_top10 = tweets_word_count.most_common(10)
    
    output = []
    for i in range(0, len(tweets_word_count_top10)):
        output.append([i+1, ', ', tweets_word_count_top10[i][0], '\n'])
    
    output = sum(output, [])
    output = ListToStr(output)

    # save to file
    result = {}
    result['date'] = date
    result['time'] = time
    result['word_count'] = tweets_word_count_top10
    file_name = date + '_top10_words.json'
    Save_Result_to_File(result,file_name)

    # save file to s3
    #local_file = '/tmp/Data/' + file_name
    #s3_file = os.path.join('Data_Analysis/top10_words', month, file_name)
    #Write_to_s3(local_file, s3_file)

    return output


## 10 most popular hashtag
def Tweets_Analysis_hashtag_top10(tweets_text_split):

    hashtag_list = []

    for w in tweets_text_split:
        w.replace(r'\u','')
        if '#' in w[0] and len(w) > 1 and w[-1] != r",":
            hashtag_list.append(w)

    hashtag_top10 = Counter(hashtag_list).most_common(10)
    publish = Format_Counter_to_Publish(hashtag_top10)

    # save to file
    result = {}
    result['date'] = date
    result['time'] = time
    result['hashtag_top10'] = hashtag_top10
    file_name = date + '_top10_hashtag.json'
    Save_Result_to_File(result,file_name)

    # save file to s3
    #local_file = '/tmp/Data/' + file_name
    #s3_file = os.path.join('Data_Analysis/top10_hashtag', month, file_name)
    #Write_to_s3(local_file, s3_file)

    return publish


## most frequent mentioned usernames
def Tweets_Analysis_mtdusr_top10(tweets_text_split):

    usernames_list = []

    for w in tweets_text_split:
        if '@' in w[0]:
            usernames_list.append(w)

    usernames_mentioned_top10 = Counter(usernames_list).most_common(10)

    # save to file
    result = {}
    result['date'] = date
    result['time'] = time
    result['usernames_mentioned_top10'] = usernames_mentioned_top10
    file_name = date + 'usernames_mentioned_top10.json'
    Save_Result_to_File(result,file_name)

    # save file to s3
    #local_file = '/tmp/Data/' + file_name
    #s3_file = os.path.join('Data_Analysis/top10_mentioned_user/', month, file_name)
    #Write_to_s3(local_file, s3_file)

    return usernames_mentioned_top10


## most common sources
def Tweets_Analysis_sources_top10(tweets_info):

    source_list = []

    for i in range(0, len(tweets_info['tweets'])):
        source_list.append(tweets_info['tweets'][i]['source'])

    source_top10 = Counter(source_list).most_common(10)
    publish = Format_Counter_to_Publish(source_top10)

    # save to file
    result = {}
    result['date'] = date
    result['time'] = time
    result['source_top10'] = source_top10
    file_name = date + 'source_top10.json'
    Save_Result_to_File(result,file_name)

    # save file to s3
    #local_file = '/tmp/Data/' + file_name
    #s3_file = os.path.join('Data_Analysis/top10_source', month, file_name)
    #Write_to_s3(local_file, s3_file)

    return publish



## most influential tweets
def Tweets_Analysis_influential_tweets_top10(tweets_info):

    tweets_influ = []

    for i in range(0, len(tweets_info['tweets'])):
        t_v = 0
        for j in tweets_info['tweets'][i]['public_metrics']:
            t_v += tweets_info['tweets'][i]['public_metrics'][j]
        item_new = [t_v, tweets_info['tweets'][i]['author_id'], tweets_info['tweets'][i]['text'], tweets_info['tweets'][i]['id']]
        tweets_influ.append(item_new)

    tweets_influ.sort(reverse=True)
    tweets_influ_top10 = tweets_influ[0:10]

    for i in range(0, len(tweets_influ_top10)):
        author_id = tweets_influ_top10[i][1]
        tweets_influ_top10[i][1] = '@' + get_username(author_id)

    df_tweets_influ_top10 = pd.DataFrame(tweets_influ_top10)
    df_tweets_influ_top10.columns = ['score', 'username','tweet','tweet_id']

    rank = [1,2,3,4,5,6,7,8,9,10]
    df_tweets_influ_top10['rank'] = rank

    json_tweets_influ_top10 = df_tweets_influ_top10.to_json()

    # save to file
    result = {}
    result['date'] = date
    result['time'] = time
    result['df_tweets_influ_top10'] = json_tweets_influ_top10
    file_name = date + '_tweets_influ_top10.json'
    Save_Result_to_File(result,file_name)

    # save file to s3
    #local_file = '/tmp/Data/' + file_name
    #s3_file = os.path.join('Data_Analysis/top10_influ_tweets', month, file_name)
    #Write_to_s3(local_file, s3_file)

    return df_tweets_influ_top10   


## Get author info function 
# cite from TwitterCollector by Gene Moo Lee, Jaecheol Park and Xiaoke Zhang
def get_username(author_id):
    username = client.get_user(id = author_id).data.username
    return username


## word cloud
def Word_Cloud(tweets_text_tokenize):

    tweets_text_str = ' '.join(tweets_text_tokenize)
    wordcloud = WordCloud(collocations=False, width=800, height=400).generate(tweets_text_str)

    if os.name == "posix":
        os.chdir('/tmp')
    else:
        os.chdir(ROOT)

    if not os.path.exists(os.path.join('Data')):
        os.makedirs('Data')

    path = os.path.join(os.getcwd(), 'Data')
    os.chdir(path)

    file_name = date + '_Word_Cloud.png'

    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(file_name)

    # save file to s3
    #local_file = '/tmp/Data/' + file_name
    #s3_file = os.path.join('Data_Analysis/Word_Cloud', month, file_name)
    #Write_to_s3(local_file, s3_file)

    plt.show()



## clean data for sentimate
def Tweets_Sentiment_Clean_Data(tweets_info):

    tweets_senti = []

    for i in range(0, len(tweets_info['tweets'])):
        tweet = tweets_info['tweets'][i]['text']
        tweet_tb = TextBlob(tweet)
        tweet_pol = tweet_tb.sentiment.polarity
        tweet_sub = tweet_tb.sentiment.subjectivity
        author_id = tweets_info['tweets'][i]['author_id']
        tweets_senti.append([tweet, tweet_pol, tweet_sub, author_id])
    
    df_tweets_senti = pd.DataFrame(tweets_senti)
    df_tweets_senti.columns = ['tweet', 'polarity', 'subjectivity', 'author_id']

    return df_tweets_senti


## what are the average polarity?
def Tweets_Analysis_sentiment_avg(tweets_info):

    df_tweets_senti = Tweets_Sentiment_Clean_Data(tweets_info)

    tweets_pol_mean = np.mean(df_tweets_senti['polarity'])
    tweets_sub_mean = np.mean(df_tweets_senti['subjectivity'])

    return tweets_pol_mean, tweets_sub_mean


## what are the most happy and sadness tweets?
def Tweets_Analysis_sentiment_top10(df_tweets_senti):

    df_tweets_ord = df_tweets_senti.sort_values(by = ['polarity'], ascending=  False)

    df_tweets_ord_top10 = df_tweets_ord[:10]

    tweets_ord_top10_username = []

    for i in range(0, len(df_tweets_ord_top10)):
        author_id = df_tweets_ord_top10.iloc[i, 3]
        tweets_ord_top10_username.append('@' + get_username(author_id))
    
    df_tweets_ord_top10['username'] = tweets_ord_top10_username

    rank = [1,2,3,4,5,6,7,8,9,10]
    df_tweets_ord_top10['rank'] = rank

    df_tweets_ord_buttom10 = df_tweets_ord[-11:-1].loc[::-1]

    tweets_ord_buttom10_username = []

    for i in range(0, len(df_tweets_ord_buttom10)):
        author_id = df_tweets_ord_buttom10.iloc[i,3]
        tweets_ord_buttom10_username.append('@' + get_username(author_id))

    df_tweets_ord_buttom10['username'] = tweets_ord_top10_username
    df_tweets_ord_buttom10['rank'] = rank


    json_tweets_ord_top10 = df_tweets_ord_top10.to_json()
    json_tweets_ord_buttom10 = df_tweets_ord_buttom10.to_json()


    # save to file
    result = {}
    result['date'] = date
    result['time'] = time
    result['df_tweets_ord_top10'] = json_tweets_ord_top10
    result['df_tweets_ord_buttom10'] = json_tweets_ord_buttom10

    file_name = date + 'df_tweets_influ_top10.json'
    Save_Result_to_File(result,file_name)

    # save file to s3
    #local_file = '/tmp/Data/' + file_name
    #s3_file = os.path.join('Data_Analysis/top10_sentiment_tweets', month, file_name)
    #Write_to_s3(local_file, s3_file)

    return df_tweets_ord_top10, df_tweets_ord_buttom10



## create tweets
def Tweets_Create(text):

    while len(text) > 280:
        tweet_1 = text[:280]
        client.create_tweet(text=tweet_1)
        text = text[280:]

    client.create_tweet(text=text)

    print("tweet published!")


## save result to file

def Save_Result_to_File(result,file_name):

    if os.name == "posix":
        os.chdir('/tmp')
    else:
        os.chdir(ROOT)


    if not os.path.exists(os.path.join('Data')):
        os.makedirs('Data')

    path = os.path.join(os.getcwd(), 'Data', file_name)

    with open(path, 'w', encoding = 'utf-8') as f:
        f.write(json.dumps(result, indent=4))
  


## List to Str
def ListToStr(list):
    str1 = ''

    str1 = ''.join([str(elem) for elem in list])

    return str1


def get_tweet(tweets_file, excluded_tweets=None):
    #Get tweet to post from CSV file

    with open(tweets_file) as csvfile:
        reader = csv.DictReader(csvfile)
        possible_tweets = [row["tweet"] for row in reader]

    if excluded_tweets:
        recent_tweets = [status_object.text for status_object in excluded_tweets]
        possible_tweets = [tweet for tweet in possible_tweets if tweet not in recent_tweets]

    selected_tweet = random.choice(possible_tweets)

    return selected_tweet



# function 'Write_to_s3' is cited from:
# https://medium.com/geekculture/how-to-upload-file-to-s3-using-python-aws-lambda-9aa03bb2c752

def Write_to_s3(local_file, s3_file):

    s3 = boto3.client('s3')

    try:
        s3.upload_file(local_file, os.environ['BUCKET_NAME'], s3_file)
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': os.environ['BUCKET_NAME'],
                'Key': s3_file
            },
            ExpiresIn=24 * 3600
        )

        print("Upload Successful!!!")
        return url
    except FileNotFoundError:
        print("The file was not found")
        return None


# read s3 files
def Read_s3_file(s3_file):
    s3 = boto3.client('s3')
    file_content = s3.get_object('BUCKET_NAME', key = s3_file)["Body"].read()
    print(file_content)


def Format_Counter_to_Publish(top_list):
    publish = []
    for i in range(0, len(top_list)):
        publish.append([i+1, ', ', top_list[i][0], '\n'])
    
    publish = sum(publish, [])
    publish = ListToStr(publish)
    return publish