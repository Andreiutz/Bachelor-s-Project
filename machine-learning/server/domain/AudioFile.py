from datetime import datetime
import json

class AudioFile:
    def __init__(self, audio_id : str, audio_name : str, last_edited : datetime, duration : float):
        self.__audio_id = audio_id
        self.__audio_name = audio_name
        self.__last_edited = last_edited
        self.__duration = duration

    def get_audio_id(self):
        return self.__audio_id

    def get_audio_name(self):
        return self.__audio_name

    def get_last_edited(self):
        return self.__last_edited

    def get_duration(self):
        return self.__duration