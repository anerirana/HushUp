import pandas as pd
import numpy as np
import re
import contractions

# Future Work :-
# Convert unicode emoticons to usable format
# Check what special charcacter set is in the corpus and decide what to keep
# Remove everything excpet alphabet and required special charcters

class TweetCleaner():

    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def start_cleaning(self):
        self.noisy_tweets = pd.read_csv(self.input_file)

        # Drop rows with NaN values
        self.noisy_tweets.dropna(axis = 0, inplace = True)
        
        # Remove re-twittes followed by user handles (RT @user:)
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: self.remove_pattern("RT @[\w]*:?", str(text)))

        # Remove remaining user handles (@user:)
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: self.remove_pattern("@[\w]*:?", text))

        # Remove urls
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: self.remove_pattern("https?://\S+", text))         

        # Expand contractions
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: contractions.fix(text))

        # Correct symbols in tweets
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: text.replace("&amp;","and"))
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: text.replace("&gt;"," "))
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: text.replace("&lt"," "))

        # Remove everything except alphabets, question mark & excalamtion mark (see if emoticons are removed safely)
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: self.remove_pattern("[^a-zA-Z?!]", text))
        # exclusion set : [^a-zA-Z0-9#']
        # all punctuation set : [!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]

        # Convert all the alphabets to lower case
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: text.lower())

        # Remove all the extra spaces left after data cleaning
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: re.sub(' +',' ',text))

        # Remove the leading and trailing spaces as well
        self.noisy_tweets['tweet'] = self.noisy_tweets['tweet'].apply(lambda text: text.strip())

        # Comment to check how much data has been dropped
        # dropped_data = self.noisy_tweets[self.noisy_tweets['tweet'].apply(lambda text: len(text) == 0)]
        # print("Total Data dropped: ", dropped_data.shape)
        # dropped_data = dropped_data[dropped_data['class'] == '1']
        # print("Offensive tweets dropped: ", dropped_data.shape)
        # print(dropped_data)

        # Drop empty rows
        self.noisy_tweets = self.noisy_tweets[self.noisy_tweets['tweet'].apply(lambda text: len(text) != 0)]
        
        self.noisy_tweets.to_csv(self.output_file, index=False)

    def remove_pattern(self, pattern, input_txt):
        def replace(match):
            return " "
        return re.sub(pattern, replace, input_txt)

dc = TweetCleaner("olid_labeled_tweets.csv", "olid_processed_tweets.csv")
dc.start_cleaning()