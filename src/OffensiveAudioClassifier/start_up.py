# Store all audios filenames and labels
# Convert all audio from speech to text and store
# Get speech features for all audios
# Data file, file_name, label, converted_text
# Get text features for all audios
# Final model


# 1 Dense, 1 dropout, 1 Dense and 1 linear (softmax)

import pandas as pd
from audio_processor import AudioProcessor
from text_processor import TextProcessor


METADATA_FILE_PATH = '../../Data/compiled_data.csv'
df = pd.read_csv(METADATA_FILE_PATH)
print("Data Loaded")
print(df.shape)

tp = TextProcessor(df['Text'])
text_df = tp.get_sentence_embeddings()
print("Text Embeddings")
print(text_df.shape)

ap = AudioProcessor()
emotion_df = ap.get_emotion_attributes()
print("Emotion Embeddings")
print(emotion_df.shape)