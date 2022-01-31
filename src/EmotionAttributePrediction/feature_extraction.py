import os
import glob
import numpy as np

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures as mtf

# Extract features from audios in the given directory and store in audio_feature_file csv file
def f1_feature_extract_and_store(dirName, audio_feature_file):
    mt_win = 1.0 
    mt_step = 1.0
    st_win = 0.05
    st_step = 0.05
    
    [all_mt_fts, wav_file_list] = f1_dirWavFeatureExtraction(dirName, mt_win, mt_step, st_win, st_step)

    wav_file_list2 = []
    for file_path in wav_file_list:
        path = file_path.split("/")
        file_name = path[len(path)-1]
        wav_file_list2.append(file_name)
    wav_file_list = wav_file_list2

    wav_file_list = np.asarray(wav_file_list)
    wav_file_list = wav_file_list.reshape(1,-1)
    wav_file_list = wav_file_list.transpose()
    rows = np.hstack((wav_file_list, all_mt_fts))

    print("shape of final array: {}".format(rows.shape))
    np.savetxt(audio_feature_file, rows, delimiter=",", fmt= "%s")

# Extract features from audios in the given directory and store in audio_feature_file csv file
def f2_feature_extract_and_store(dirName, audio_feature_file):
    mt_win = 1.0 
    mt_step = 1.0
    st_win = 0.05
    st_step = 0.05
    
    [all_mt_fts, wav_file_list, mt_ft_names] = f2_dirWavFeatureExtraction(dirName, mt_win, mt_step, st_win, st_step)

    wav_file_list2 = []
    for file_path in wav_file_list:
        path = file_path.split("/")
        file_name = path[len(path)-1]
        wav_file_list2.append(file_name)
    wav_file_list = wav_file_list2

    wav_file_list = np.asarray(wav_file_list)
    wav_file_list = wav_file_list.reshape(1,-1)
    wav_file_list = wav_file_list.transpose()
    rows = np.hstack((wav_file_list, all_mt_fts))
 
    mt_ft_names = np.asarray(mt_ft_names)
    mt_ft_names = mt_ft_names.reshape(1,-1)
    col_names = np.insert(mt_ft_names, 0, "file_name")
    array = np.vstack((col_names, rows))

    print("shape of final array: {}".format(array.shape))
    np.savetxt(audio_feature_file, array, delimiter=",", fmt= "%s")

def f1_dirWavFeatureExtraction(dirName, mt_win, mt_step, st_win, st_step):
    """
    This function extracts the mid-term features of the WAVE files of a 
    particular folder.
    The resulting feature vector is concatenated and padded to cover 
    10 seconds of the WAV file.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.
    ARGUMENTS:
        - dirName:        the path of the WAVE directory
        - mt_win, mt_step:    mid-term window and step (in seconds)
        - st_win, st_step:    short-term window and step (in seconds)
    """
    all_mt_feats = np.array([])
    wav_file_list = glob.glob(dirName+'*.wav')
    
    wav_file_list = sorted(wav_file_list)    
    wav_file_list2 = []

    for i, wavFile in enumerate(wav_file_list):  
        if os.stat(wavFile).st_size == 0:
            print("   (EMPTY FILE -- SKIPPING)")
            continue    

        [fs, x] = audioBasicIO.read_audio_file(wavFile)
        if isinstance(x, int):
            continue        
       
        x = audioBasicIO.stereo_to_mono(x)
        if x.shape[0]<float(fs)/5:
            print("  (AUDIO FILE TOO SMALL - SKIPPING)")
            continue

        wav_file_list2.append(wavFile)
        [mt_term_feats, _, _] = \
                mtf.mid_feature_extraction(x, fs, round(mt_win * fs),
                                            round(mt_step * fs),
                                            round(fs * st_win), 
                                            round(fs * st_step))

        mt_term_feats = np.transpose(mt_term_feats)

        # bring all mt features to same size (10 sec audio length)
        if mt_term_feats.shape[0] < 10:
            # pad zeros
            arr_size = mt_term_feats.shape[1]
            num_arr = 10 - mt_term_feats.shape[0]
            padding = np.zeros(arr_size*num_arr)
            mt_term_feats = mt_term_feats.flatten()
            mt_term_feats = np.hstack((mt_term_feats, padding))
        elif mt_term_feats.shape[0] > 10:
            # truncate
            mt_term_feats = mt_term_feats[:10][:]
            mt_term_feats = mt_term_feats.flatten()
        else:
            mt_term_feats = mt_term_feats.flatten()

        if (not np.isnan(mt_term_feats).any()) and \
                (not np.isinf(mt_term_feats).any()):
            if len(all_mt_feats) == 0:
                # append feature vector
                all_mt_feats = mt_term_feats
            else:
                all_mt_feats = np.vstack((all_mt_feats, mt_term_feats))
    return (all_mt_feats, wav_file_list2)


def f2_dirWavFeatureExtraction(dirName, mt_win, mt_step, st_win, st_step):
    """
    This function extracts the mid-term features of the WAVE files of a 
    particular folder.
    The resulting feature vector is extracted by long-term averaging the
    mid-term features.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.
    ARGUMENTS:
        - dirName:        the path of the WAVE directory
        - mt_win, mt_step:    mid-term window and step (in seconds)
        - st_win, st_step:    short-term window and step (in seconds)
    """
    all_mt_feats = np.array([])
    wav_file_list = glob.glob(dirName+'*.wav')
    
    wav_file_list = sorted(wav_file_list)    
    wav_file_list2, mt_feature_names = [], []

    for i, wavFile in enumerate(wav_file_list):  
        if os.stat(wavFile).st_size == 0:
            print("   (EMPTY FILE -- SKIPPING)")
            continue    

        [fs, x] = audioBasicIO.read_audio_file(wavFile)
        if isinstance(x, int):
            continue        
       
        x = audioBasicIO.stereo_to_mono(x)
        if x.shape[0]<float(fs)/5:
            print("  (AUDIO FILE TOO SMALL - SKIPPING)")
            continue
        
        wav_file_list2.append(wavFile)
        [mt_term_feats, _, mt_feature_names] = \
                mtf.mid_feature_extraction(x, fs, round(mt_win * fs),
                                            round(mt_step * fs),
                                            round(fs * st_win), 
                                            round(fs * st_step))

        mt_term_feats = np.transpose(mt_term_feats)
    
        # long term averaging of mid-term statistics
        mt_term_feats = mt_term_feats.mean(axis=0)

        if (not np.isnan(mt_term_feats).any()) and \
                (not np.isinf(mt_term_feats).any()):
            if len(all_mt_feats) == 0:
                # append feature vector
                all_mt_feats = mt_term_feats
            else:
                all_mt_feats = np.vstack((all_mt_feats, mt_term_feats))
    return (all_mt_feats, wav_file_list2, mt_feature_names)

dirName = "/HushUp/Data/Datasets/*/"
audio_feature_file = "audio_features.csv"
f2_feature_extract_and_store(dirName, audio_feature_file)