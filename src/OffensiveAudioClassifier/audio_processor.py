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

    def get_emotion_attributes(self):
        emotion_df = pd.DataFrame(columns = ['Audio_name', 'Valence_attributes', 'Arousal_attributes', 'Dominance_attributes'])
        valence_model, arousal_model, dominance_model = self.load_models()
        wav_filepath_list = glob.glob(self.audio_dir +'*.wav', recursive = True)
        audio_list, filenames_list = self.convert_wav_to_timeseries(wav_filepath_list)
        audio_list = np.array(audio_list)
        # print(audio_list.shape)
        with open(self.working_dir +'/audio_timeseries.pkl', 'wb') as f:
            pickle.dump(audio_list, f)
        with open(self.working_dir +'/filenames.pkl', 'wb') as f:
            pickle.dump(filenames_list, f)
    
        valence_attributes_list  = valence_model.predict(audio_list)
        # print(valence_attributes_list.shape)
        with open(self.working_dir +'/valence_attributes.pkl', 'wb') as f:
            pickle.dump(valence_attributes_list, f)
        valence_attributes=[]
        for i in range(0, len(valence_attributes_list)):
            valence_attributes.append(valence_attributes_list[i])
        emotion_df['Valence_attributes'] = valence_attributes

        arousal_attributes_list  = arousal_model.predict(audio_list)
        # print(arousal_attributes_list.shape)
        with open(self.working_dir +'/arousal_attributes.pkl', 'wb') as f:
            pickle.dump(arousal_attributes_list, f)
        arousal_attributes=[]
        for i in range(0, len(arousal_attributes_list)):
            arousal_attributes.append(arousal_attributes_list[i])
        emotion_df['Arousal_attributes'] = arousal_attributes

        dominance_attributes_list  = dominance_model.predict(audio_list)
        # print(dominance_attributes_list.shape)
        with open(self.working_dir +'/dominance_attributes.pkl', 'wb') as f:
            pickle.dump(dominance_attributes_list, f)
        dominance_attributes=[]
        for i in range(0, len(dominance_attributes_list)):
            dominance_attributes.append(dominance_attributes_list[i])
        emotion_df['Dominance_attributes'] = dominance_attributes
        #store_individual_pickles(audio_list, filenames_list, valence_model, arousal_model, dominance_model)
        return emotion_df
