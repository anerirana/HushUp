#!/usr/bin/env python3
import speech_recognition as sr
import logging
import os
import pandas as pd
import time
import glob

def convert_and_store():
	file_names = []
	sentences = []
	labels = []

	dirName = "/HushUp/Data/Datasets/*/"
	file_list = glob.glob(dirName+'*.wav')

	for file in file_list:
		# Check if it is wav file format
		if file.name.endswith(".wav"):
			try:
				logging.debug("Processing " + file.name)
				sentence = convert_speech_to_text(file.name)
				label = get_file_label(file.name, sentence)
				sentences.append(sentence)
				labels.append(label)
				file_names.append(file.name)
				time.sleep(2)
			except:
				logging.error("Could not process file " + file.name)
				break


	df = pd.DataFrame({"file_name": file_names, "sentence": sentences, "label": labels})

    # Store extracted text
	df.to_csv("/HushUp/Data/text_data.csv", index = None)

def convert_speech_to_text(AUDIO_FILE):
	# use the audio file as the audio source
	r = sr.Recognizer()
	with sr.AudioFile(AUDIO_FILE) as source:
		audio = r.record(source) # read the entire audio file

	# recognize speech using Houndify
	HOUNDIFY_CLIENT_ID = "" 
	HOUNDIFY_CLIENT_KEY = "" 

	try:
		houndify_speech = r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY)
		logging.debug("\nHoundify thinks you said " + houndify_speech)
		return houndify_speech
	except sr.UnknownValueError:
		logging.error("\nHoundify could not understand audio")
		return None
	except sr.RequestError as e:
		logging.error("Could not request results from Houndify service; {0}".format(e))
		return None
	except:
		logging.error("Exception ocurred while fetching from Houndify !!")
		return None

def get_file_label(file_name, sentence):
    if sentence != None:
        name_components = file_name.split("_")
        if name_components[1] == "nonoffensive":
            label = "0"
        elif name_components[1] == "offensive":
            label = "1"
        else:
            label = None
    else:
        label = None
    return label

convert_and_store() 