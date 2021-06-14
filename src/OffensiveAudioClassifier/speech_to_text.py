#!/usr/bin/env python3

import pandas as pd
import speech_recognition as sr
import sys
import os

def convert_speech_to_text(AUDIO_FILE):
	# use the audio file as the audio source
	r = sr.Recognizer()
	choice = 0
	with sr.AudioFile(AUDIO_FILE) as source:
	    audio = r.record(source) # read the entire audio file

	# Alternate credentials if first one is exhausted
	# HOUNDIFY_CLIENT_ID = "JdxYfWqMU4NDy5nWFQtlFA==" 
	# HOUNDIFY_CLIENT_KEY = "02Nel0FkAxH5WkgZQwPqCJ8fu5krfTqploKjakMXiWkCe5WCLBDJVsiBn0s_CYswUoigqBXtRw_QXmcRDBP7Gw==" 

	# recognize speech using Houndify
	HOUNDIFY_CLIENT_ID = "xVNe64h8Rd8BoGhxLAwpHA==" 
	HOUNDIFY_CLIENT_KEY = "7FH_O-rFM0-VTBmBck4hIFhqaPOGe-9-7HWvMDXwlutlJrRzGF4XTAI0vqFULHCQvTpkRCfJuodmQ_b3O_ad6Q==" 
	try:
		houndify_speech = r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY)
		# to-do: logging
		# print("\nHoundify thinks you said " + houndify_speech)
		return houndify_speech
	except sr.UnknownValueError:
		# to-do: logging
		print("\nHoundify could not understand audio")
		return None
	except sr.RequestError as e:
		# to-do: logging
		print("Could not request results from Houndify service; {0}".format(e))
		return None
	except:
		# to-do: logging
		print("Exception ocurred while fetching from Houndify !!")
		return None

# to-do:
# Read filename from csv
# Get audio from '../../Data/Audios'
# append text to list
# concat dfs
# write to csv

print(convert_speech_to_text('../../Data/Datasets/offensive_yt_1to130/1_00001.wav'))
