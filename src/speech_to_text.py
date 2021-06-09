#!/usr/bin/env python3

import speech_recognition as sr
import sys
import os

def convert_speech_to_text(AUDIO_FILE, converter):
	# use the audio file as the audio source
	r = sr.Recognizer()
	choice = 0
	with sr.AudioFile(AUDIO_FILE) as source:
	    audio = r.record(source) # read the entire audio file

	if converter == "witai":
		# recognize speech using wit ai software
		WIT_AI_KEY = "T344FT6KQDJMOW3GCZLCKAPI7C4U2ILT" 
		try:
			wit_speech = r.recognize_wit(audio, key=WIT_AI_KEY)
			print("\nWit.ai thinks you said " + wit_speech)
		except sr.UnknownValueError:
			print("\nWit.ai could not understand audio")
		except sr.RequestError as e:
			print("Could not request results from WITAI service; {0}".format(e))
	
	elif converter == "houndify":
		# recognize speech using Houndify
		HOUNDIFY_CLIENT_ID = "JdxYfWqMU4NDy5nWFQtlFA==" 
		HOUNDIFY_CLIENT_KEY = "02Nel0FkAxH5WkgZQwPqCJ8fu5krfTqploKjakMXiWkCe5WCLBDJVsiBn0s_CYswUoigqBXtRw_QXmcRDBP7Gw==" 
		try:
			houndify_speech = r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY)
			print("\nHoundify thinks you said " + houndify_speech)
			return houndify_speech
		except sr.UnknownValueError:
			print("\nHoundify could not understand audio")
			return None
		except sr.RequestError as e:
			print("Could not request results from Houndify service; {0}".format(e))
			return None
		except:
			print("Exception ocurred while fetching from Houndify !!")
			return None

def convert_speech_to_text2(AUDIO_FILE):
	# use the audio file as the audio source
	r = sr.Recognizer()
	choice = 0
	with sr.AudioFile(AUDIO_FILE) as source:
	    audio = r.record(source) # read the entire audio fileprint("Could not request results from Wit.ai service; {0}".format(e))

	# recognize speech using Houndify
	HOUNDIFY_CLIENT_ID = "xVNe64h8Rd8BoGhxLAwpHA==" 
	HOUNDIFY_CLIENT_KEY = "7FH_O-rFM0-VTBmBck4hIFhqaPOGe-9-7HWvMDXwlutlJrRzGF4XTAI0vqFULHCQvTpkRCfJuodmQ_b3O_ad6Q==" 
	try:
		houndify_speech = r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY)
		print("\nHoundify thinks you said " + houndify_speech)
		return houndify_speech
	except sr.UnknownValueError:
		print("\nHoundify could not understand audio")
	except sr.RequestError as e:
		print("Could not request results from Houndify service; {0}".format(e))

#filename = sys.argv[1]
#os.system("aplay " + filename)
#print(convert_speech_to_text(filename))
