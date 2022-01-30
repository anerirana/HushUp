import glob
import os
import librosa
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential

class AudioProcessor():

    def __init__(self):
        self.working_dir = '../../Data/EmotionAttributes'
        self.audio_dir = '../../Data/AudioSet/'
        self.models_dir = '../AudioEmotionClassifier/Models'
        self.audio_length = 35000
        self.target_sr = 8000

   
    def split_filename(self, filename):
        split_string= filename.split(sep='.')
        return split_string[0]


    def convert_wav_to_timeseries(self, wav_filepath_list):
        filenames_list = []
        audio_list = []
        for wav_filepath in wav_filepath_list:
            wav_filename  = wav_filepath.split(sep= '/')
            filename = self.split_filename(wav_filename[-1])
            filenames_list.append(filename)
            audio_buf, _ = librosa.load(wav_filepath, sr=self.target_sr, mono=True)
            audio_buf = audio_buf.reshape(-1, 1)
            audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
            original_length = len(audio_buf)
            if original_length < self.audio_length:
                audio_buf = np.concatenate((audio_buf, np.zeros(shape=(self.audio_length - original_length, 1))))
            elif original_length > self.audio_length:
                audio_buf = audio_buf[0:self.audio_length]
            audio_list.append(audio_buf)
        return audio_list, filenames_list

    def load_models(self):
        valence_model = keras.models.load_model(self.models_dir +'/valence_model.h5')
        valence_model.pop()

        arousal_model = keras.models.load_model(self.models_dir +'/arousal_model.h5')
        arousal_model.pop()

        dominance_model = keras.models.load_model(self.models_dir +'/dominance_model.h5')
        dominance_model.pop()

        return valence_model, arousal_model, dominance_model

    def store_individual_pickles(self, audio_list, filenames_list, valence_model, arousal_model, dominance_model):
        audio_list = np.array(audio_list)
        # print(audio_list.shape)
        for i in range(len(audio_list)):
            filename = filenames_list[i]
            valences = valence_model.predict(audio_list[i])
            arousals = arousal_model.predict(audio_list[i])
            dominances = dominance_model.predict(audio_list[i])
            all_emotion_attributes = np.concatenate((valences, arousals, dominances), axis=0)
            with open(self.working_dir +'/filewise_all_attriutes/'+filename+'.pkl', 'wb') as f:
                pickle.dump(all_emotion_attributes, f)

    def get_emotion_attributes(self, audio_list):
        emotion_df = pd.DataFrame()
        valence_model, arousal_model, dominance_model = self.load_models()

        wav_filepath_list = []
        for file_name in audio_list:
            wav_filepath_list.append(self.audio_dir + file_name + '.wav')
        audio_list, filenames_list = self.convert_wav_to_timeseries(wav_filepath_list)
        audio_list = np.array(audio_list)
        with open(self.working_dir +'/audio_timeseries.pkl', 'wb') as f:
            pickle.dump(audio_list, f)
        with open(self.working_dir +'/filenames.pkl', 'wb') as f:
            pickle.dump(filenames_list, f)
    
        valence_attributes_list  = valence_model.predict(audio_list)
        # print(valence_attributes_list.shape)
        with open(self.working_dir +'/valence_attributes.pkl', 'wb') as f:
            pickle.dump(valence_attributes_list, f)
        valence_df = pd.DataFrame(valence_attributes_list)

        arousal_attributes_list  = arousal_model.predict(audio_list)
        # print(arousal_attributes_list.shape)
        with open(self.working_dir +'/arousal_attributes.pkl', 'wb') as f:
            pickle.dump(arousal_attributes_list, f)
        arousal_df = pd.DataFrame(arousal_attributes_list)

        dominance_attributes_list  = dominance_model.predict(audio_list)
        # print(dominance_attributes_list.shape)
        with open(self.working_dir +'/dominance_attributes.pkl', 'wb') as f:
            pickle.dump(dominance_attributes_list, f)
        dominance_attributes=[]
        dominance_df = pd.DataFrame(dominance_attributes_list)
        

        emotion_df = pd.concat([emotion_df, valence_df, arousal_df, dominance_df], axis=1)
        #store_individual_pickles(audio_list, filenames_list, valence_model, arousal_model, dominance_model)
        return emotion_df
