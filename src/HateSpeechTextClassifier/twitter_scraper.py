import tweepy
from tweepy import RateLimitError
from tweepy import TweepError
import csv
import time
import sys

# Twitter Developer keys here
consumer_key = ''
consumer_key_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# This method creates the training set
def createTrainingSet(corpusFile, targetResultFile, invalidTweetsFile):

    corpus = []
    original_tweet_data = []

    with open(corpusFile, 'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',')
        for row in lineReader:
            corpus.append({"tweet_id": row[0], "label": row[1]})
            original_tweet_data.append([row[0],row[1]])

    sleepTime = 2
    trainingDataSet = []
    invalid_tweet_ids = []

    for tweet in corpus:
        try:
            tweetFetched = api.get_status(tweet["tweet_id"], tweet_mode = "extended")
            print("Tweet fetched " + tweetFetched.full_text)
            tweet["text"] = tweetFetched.full_text
            trainingDataSet.append(tweet)
            original_tweet_data.remove([tweet["tweet_id"],tweet["label"]])
            time.sleep(sleepTime)

        except RateLimitError:
            print("RateLimitError !!")
            print("Try after 1 hr")
            break

        except TweepError as err:

            # Error code 144 means that tweet is no longer available, move to next tweet
            if err.args[0][0]['code'] == 144:
                print("Tweet ID " + tweet["tweet_id"] + " does not exist")
                invalid_tweet_ids.append(tweet["tweet_id"])
                original_tweet_data.remove([tweet["tweet_id"],tweet["label"]])
                continue

            # Error code 63 means that user has been suspended, move to next tweet
            elif err.args[0][0]['code'] == 63:
                print("User with Tweet ID " + tweet["tweet_id"] + " has been suspended")
                invalid_tweet_ids.append(tweet["tweet_id"])
                original_tweet_data.remove([tweet["tweet_id"],tweet["label"]])
                continue

            # Error code 179 means that user's privacy settings does not allow you extract this tweet, move to next tweet
            elif err.args[0][0]['code'] == 179:
                print("Not authenticated to extract Tweet ID " + tweet["tweet_id"])
                invalid_tweet_ids.append(tweet["tweet_id"])
                original_tweet_data.remove([tweet["tweet_id"],tweet["label"]])
                continue

            # Error code 34 means that page does not exist, move to next tweet
            elif err.args[0][0]['code'] == 34:
                print("Page corresponding to tweet ID " + tweet["tweet_id"] + " does not exist")
                invalid_tweet_ids.append(tweet["tweet_id"])
                original_tweet_data.remove([tweet["tweet_id"],tweet["label"]])
                continue

            else:
                print("Unkown TweepError:", err)
                break

        except:
            print("Unkown Exception:", sys.exc_info()[0])
            break

    with open(corpusFile, 'w') as csvfile:
        linewriter = csv.writer(csvfile, delimiter=',')
        for tweet in original_tweet_data:
            try:
                linewriter.writerow(tweet)
            except Exception as e:
                print(e)

    with open(targetResultFile, 'a') as csvfile:
        linewriter = csv.writer(csvfile, delimiter=',')
        for tweet in trainingDataSet:
            try:
                linewriter.writerow([tweet["tweet_id"], tweet["label"], tweet["text"]])
            except Exception as e:
                print(e)

    with open(invalidTweetsFile, 'a') as csvfile:
        linewriter = csv.writer(csvfile, delimiter='\n')
        try:
            linewriter.writerow(invalid_tweet_ids)
        except Exception as e:
            print(e)
    return trainingDataSet


# This is corpus dataset
corpusFile = "HushUp/Data/waseem_tweet_ids.csv"
# This is my target file
targetResultFile = "HushUp/Data/waseem_labeled_tweets.csv"
# File to store invalid tweets IDs no longer available
invalidTweetsFile = "HushUp/Data/waseem_invalid_tweets.csv"
# Call the method
resultFile = createTrainingSet(corpusFile, targetResultFile, invalidTweetsFile)