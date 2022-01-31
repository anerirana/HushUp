import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Change delimiter of a csv file
def change_delimiter():
    reader = csv.reader(open("source.csv", "rU"), delimiter=',')
    writer = csv.writer(open("target.csv", 'w'), delimiter='\n')
    writer.writerows(reader)

    print("Delimiter successfully changed")

# combine davison and waseem datasets to make a single labeled csv file
def create_dataset_A():
    waseem_dataset = pd.read_csv('waseem_labeled_tweets.csv', names = ['tweet_id', 'class', 'tweet'])
    davidson_dataset = pd.read_csv('davidson_labeled_tweets.csv', index_col=0)

    # We only care about offensive(1) and non-offensive(0)
    waseem_dataset['class'].replace('none','0', inplace=True)
    waseem_dataset['class'].replace('racism','1', inplace=True)
    waseem_dataset['class'].replace('sexism','1', inplace=True)

    # Dropping 15k rows with class "offensive but not racist" to prevent skewing
    davidson_dataset = davidson_dataset.groupby('class').head(4190)

    # labels in davidson dataset are stored as int
    # Converting int labels into str offensive(1) and non-offensive(0)
    davidson_dataset['class'].replace(0, '1', inplace=True)
    davidson_dataset['class'].replace(2, '0', inplace=True)
    davidson_dataset['class'].replace(1, '1', inplace=True)

    # Check dataset distribution
    print("Davidson dataset distribution :-")
    print(davidson_dataset.groupby('class').count())
    print("Waseem dataset distribution :-")
    print(waseem_dataset.groupby('class').count())

    # Drop all irrelevant data
    # Combine datasets with label and tweets
    davidson_dataset.drop(columns = ['count', 'hate_speech', 'offensive_language', 'neither'], inplace=True)
    waseem_dataset.drop(columns = ['tweet_id'], inplace=True)
    combined_dataset = pd.concat([davidson_dataset, waseem_dataset])

    # Check combined dataset distribution
    print("Combined dataset distribution :-")
    print(combined_dataset.groupby('class').count())

    target_file = "final_labeled_tweets.csv"
    combined_dataset.to_csv(target_file, index=False)

# New experiment with combining datasets. Drop offensive tweets from waseem dataset (becasue noisy)
def create_dataset_B():
    waseem_dataset = pd.read_csv('waseem_labeled_tweets.csv', names = ['tweet_id', 'class', 'tweet'])
    davidson_dataset = pd.read_csv('davidson_labeled_tweets.csv', index_col=0)

    # Only keep non-offensive rows
    waseem_dataset = waseem_dataset[waseem_dataset['class'] == 'none']
    waseem_dataset['class'].replace('none','0', inplace=True)

    # Dropping 15k rows with class "offensive but not racist" to prevent skewing
    davidson_dataset = davidson_dataset.groupby('class').head(4190)

    # labels in davidson dataset are stored as int
    # Converting int labels into str offensive(1) and non-offensive(0)
    davidson_dataset['class'].replace(0, '1', inplace=True)
    davidson_dataset['class'].replace(2, '0', inplace=True)
    davidson_dataset['class'].replace(1, '1', inplace=True)

    # Check dataset distribution
    print("Davidson dataset distribution :-")
    print(davidson_dataset.groupby('class').count())
    print("Waseem dataset distribution :-")
    print(waseem_dataset.groupby('class').count())

    # Drop all irrelevant data
    # Combine datasets with label and tweets
    davidson_dataset.drop(columns = ['count', 'hate_speech', 'offensive_language', 'neither'], inplace=True)
    waseem_dataset.drop(columns = ['tweet_id'], inplace=True)
    combined_dataset = pd.concat([davidson_dataset, waseem_dataset])

    # Check combined dataset distribution
    print("Combined dataset distribution :-")
    print(combined_dataset.groupby('class').count())

    target_file = "final_labeled_tweets.csv"
    combined_dataset.to_csv(target_file, index=False)

# Use for subsampling or splitting datasets
def subSampleDatsets(input_file, train_file, test_file, sampling_size):
    df = pd.read_csv(input_file)
    tweets = np.array(df['tweet'])
    classes = np.array(df['class'])
    print("Current dataset size: ", df.shape)
    print("Current dataset distribution:- \n", df.groupby(df['class']).count())

    # Split data
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(tweets, classes, random_state=2018, test_size=sampling_size)
    print("Length of training dataset: " + str(len(train_inputs)))
    print("Length of testing dataset: " + str(len(test_inputs)))

    train_df = pd.DataFrame()
    train_df['tweet'] = train_inputs
    train_df['class'] = train_labels
    print("Training dataset size: ", train_df.shape)
    print("Training dataset distribution:- \n", train_df.groupby(train_df['class']).count())
    train_df.to_csv(train_file, index=None)

    test_df = pd.DataFrame()
    test_df['tweet'] = test_inputs
    test_df['class'] = test_labels
    print("Testing dataset size: ", test_df.shape)
    print("Testing dataset distribution:- \n", test_df.groupby(test_df['class']).count())
    test_df.to_csv(test_file, index=None)


# subSampleDatsets("olid_processed_tweets.csv", "train_data.csv", "test_data.csv", 0.1)
create_dataset_A()