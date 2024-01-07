from datetime import datetime

class AudioFile:
    def __init__(self, audio_id : str, audio_name : str, user_id : str, date_posted : datetime):
        self.__audio_id = audio_id
        self.__audio_name = audio_name
        self.__user_id = user_id
        self.__date_posted = date_posted

    def get_audio_id(self):
        return self.__audio_id

    def get_audio_name(self):
        return self.__audio_name

    def get_user_id(self):
        return self.__user_id

    def get_date_posted(self):
        return self.__date_posted

