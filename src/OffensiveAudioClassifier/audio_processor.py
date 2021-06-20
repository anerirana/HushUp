import glob
import os
import librosa
import numpy as np
import pickle
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential

class AudioProcessor():

    def __init__(self, filenames_list =[], audio_list =[], audio_dir, working_dir, audio_length, target_sr, models_dir, filename_index):
        AUDIO_DIR = '../../Data/Datasets/'
        WORKING_DIR = '../../Data/Datasets/EmotionData'
        MODELS_DIR = '../AudioEmotionClassifier/Models'
        AUDIO_LENGTH = 35000
        TARGET_SR = 8000
        FILENAME_INDEX = 7

        self.working_dir = WORKING_DIR
        self.audio_dir = AUDIO_DIR
        self.filename_index = FILENAME_INDEX
        self.models_dir = MODELS_DIR
        self.audio_length = AUDIO_LENGTH
        self.target_sr = TARGET_SR

   
    def split_filename(filename):
	split_string= filename.split(sep='.')
	return split_string[0]


    def convert_wav_to_timeseries(wav_filepath_list):
        for wav_filepath in wav_filepath_list:
            wav_filename  = wav_filepath.split(sep= '/')
            filename = split_filename(wav_filename[FILENAME_INDEX])
            print(filename)
            filenames_list.append(filename)
            audio_buf, _ = librosa.load(wav_filepath, sr=TARGET_SR, mono=True)
            audio_buf = audio_buf.reshape(-1, 1)
            audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
            original_length = len(audio_buf)
            if original_length < AUDIO_LENGTH:
                audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
            elif original_length > AUDIO_LENGTH:
                audio_buf = audio_buf[0:AUDIO_LENGTH]
            audio_list.append(audio_buf)
        return audio_list, filenames_list

    def load_models():
        valence_model_whole = keras.models.load_model(models_dir +'/valence_model.h5')
        print(valence_model_whole.summary())
        valence_model = valence_model_whole.pop()
        print(valence_model.summary())

        arousal_model_whole = keras.models.load_model(models_dir +'/arousal_model.h5')
        print(arousal_model_whole.summary())
        arousal_model = arousal_model_whole.pop()
        print(arousal_model.summary())

        dominance_model_whole = keras.models.load_model(models_dir +'/dominance_model.h5')
        print(dominance_model_whole.summary())
        dominance_model = dominance_model_whole.pop()
        print(dominance_model.summary())
        return valence_model, arousal_model, dominance_model

    def store_individual_pickles(audio_list, filenames_list, valence_model, arousal_model, dominance_model):
        audio_list = np.array(audio_list)
        print(audio_list.shape)
        for i in range(len(audio_list)):
            filename = filenames_list[i]
            valences = valence_model.predict(audio_list[i])
            arousals = arousal_model.predict(audio_list[i])
            dominances = dominance_model.predict(audio_list[i])
            all_emotion_attributes = np.concatenate((valences, arousals, dominances), axis=0)
            with open(working_dir +'/EmotionAttributes/filewise_all_attriutes/'+filename+'.pkl', 'wb') as f:
                pickle.dump(all_emotion_attributes, f)

    def get_emotion_attributes():
        emotion_df = pd.DataFrame(columns = ['Audio_name', 'Valence_attributes', 'Arousal_attributes', 'Dominance_attributes'])
        valence_model, arousal_model, dominance_model = load_models()
        wav_filepath_list = glob.glob(audio_dir +'/*/*.wav', recursive = True)
        audio_list, filenames_list = convert_wav_to_timeseries(wav_filepath_list)
        audio_list = np.array(audio_list)
        print(audio_list.shape)
        with open(working_dir +'/EmotionAttributes/audio_timeseries.pkl', 'wb') as f:
            pickle.dump(audio_list, f)
        with open(working_dir +'/EmotionAttributes/filenames.pkl', 'wb') as f:
            pickle.dump(filenames_list, f)
    
        valence_attributes_list  = valence_model.predict(audio_list)
        print(valence_attributes_list.shape)
        with open(working_dir +'/EmotionAttributes/valence_attributes.pkl', 'wb') as f:
            pickle.dump(valence_attributes_list, f)
        valence_attributes=[]
        for i in range(0, len(valence_attributes_list)):
            valence_attributes.append(valence_attributes_list[i])
        emotion_df['Valence_attributes'] = valence_attributes

        arousal_attributes_list  = arousal_model.predict(audio_list)
        print(arousal_attributes_list.shape)
        with open(working_dir +'/EmotionAttributes/arousal_attributes.pkl', 'wb') as f:
            pickle.dump(arousal_attributes_list, f)
        arousal_attributes=[]
        for i in range(0, len(arousal_attributes_list)):
            arousal_attributes.append(arousal_attributes_list[i])
        emotion_df['Arousal_attributes'] = arousal_attributes

        dominance_attributes_list  = dominance_model.predict(audio_list)
        print(dominance_attributes_list.shape)
        with open(working_dir +'/EmotionAttributes/dominance_attributes.pkl', 'wb') as f:
            pickle.dump(dominance_attributes_list, f)
        dominance_attributes=[]
        for i in range(0, len(dominance_attributes_list)):
            dominance_attributes.append(dominance_attributes_list[i])
        emotion_df['Dominance_attributes'] = dominance_attributes
        #store_individual_pickles(audio_list, filenames_list, valence_model, arousal_model, dominance_model)
        return emotion_df


    def get_embeddings():
        df = get_emotion_attributes()
        return df