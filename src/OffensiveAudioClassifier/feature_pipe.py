import pandas as pd
from audio_processor import AudioProcessor
from text_processor import TextProcessor

DATA_DIR = '../../Data'
METADATA_FILE_NAME = '/compiled_data.csv'
FEATURE_FILE_NAME = '/feature_set.csv'


df = pd.read_csv(DATA_DIR + METADATA_FILE_NAME)
print("Data Loaded")
print(df.shape)

tp = TextProcessor(df['Text'])
text_df = tp.get_sentence_embeddings()
print("Text Embeddings")
print(text_df.shape)

ap = AudioProcessor()
emotion_df = ap.get_emotion_attributes(df['Filename'])
print("Emotion Embeddings")
print(emotion_df.shape)

feature_df = pd.concat([df, text_df, emotion_df], axis=1)
print(feature_df.shape)

feature_df.to_csv(DATA_DIR + FEATURE_FILE_NAME, index=False)