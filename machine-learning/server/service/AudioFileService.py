import os.path
from datetime import datetime
from server.domain.AudioFIle import AudioFile
from server.exceptions.FileNotFoundException import FileNotFoundException
import uuid

from fastapi import UploadFile

from server.exceptions.InvalidFIleFormatException import InvalidFileFormatException
from server.repository.AudioFileRepository import AudioFileRepository

class AudioFileService:
    def __init__(self, repository : AudioFileRepository):
        self.__repository = repository


    def persist_audio_file(self, file : UploadFile):
        if not file.filename.endswith('.wav'):
            raise InvalidFileFormatException('Invalid file format')

        unique_id = str(uuid.uuid4())
        original_file_name = file.filename.replace('.wav', '')
        date_posted = datetime.now()
        audio_file = AudioFile(audio_id=unique_id, audio_name=original_file_name, last_edited=date_posted)

        os.mkdir(f"../data/{unique_id}/")

        with open(f"../data/{unique_id}/file.wav", 'wb') as out_file:
            out_file.write(file.file.read())

        self.__repository.add_file(audio_file=audio_file)

        return audio_file

    def get_audio_path(self, audio_id: str) ->  str:
        file_path = f"../data/{audio_id}/file.wav"
        if not os.path.exists(file_path):
            raise FileNotFoundException("file not found")
        return file_path