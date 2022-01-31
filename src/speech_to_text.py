#!/usr/bin/env python3
import speech_recognition as sr
import logging

def convert_speech_to_text(AUDIO_FILE):
	# use the audio file as the audio source
	r = sr.Recognizer()
	choice = 0
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
