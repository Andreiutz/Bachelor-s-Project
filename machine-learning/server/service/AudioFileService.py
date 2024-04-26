import os.path
import shutil
from datetime import datetime
from server.domain.AudioFile import AudioFile
from server.exceptions.FileNotFoundException import FileNotFoundException
import uuid
import re

from fastapi import UploadFile

from server.exceptions.InvalidFIleFormatException import InvalidFileFormatException
from server.repository.AudioFileRepository import AudioFileRepository

from server.service.utils import get_audio_duration

class AudioFileService:
    def __init__(self, repository : AudioFileRepository):
        self.__repository = repository

    def persist_audio_file(self, file : UploadFile):
        if not file.filename.endswith('.wav'):
            raise InvalidFileFormatException('Invalid file format')

        original_file_name = file.filename.replace('.wav', '')
        if not self.__validate_file_name(original_file_name):
            raise Exception('Invalid file name')

        unique_id = str(uuid.uuid4())
        date_posted = datetime.now()

        os.mkdir(f"../data/{unique_id}/")
        audio_path = f"../data/{unique_id}/file.wav"

        with open(audio_path, 'wb') as out_file:
            out_file.write(file.file.read())

        audio_duration = get_audio_duration(audio_path)

        audio_file = AudioFile(audio_id=unique_id, audio_name=original_file_name, last_edited=date_posted, duration=audio_duration)
        self.__repository.add_file(audio_file=audio_file)

        return audio_file

    def get_audio_path(self, audio_id: str) ->  str:
        file_path = f"../data/{audio_id}/file.wav"
        if not os.path.exists(file_path):
            raise FileNotFoundException("file not found")
        return file_path

    def delete_audio(self, audio_id: str) -> AudioFile:
        audio_file = self.__repository.get_file(audio_id)
        if audio_file is None:
            raise FileNotFoundException("File not found")
        self.__repository.delete_file(audio_id)
        shutil.rmtree(f"../data/{audio_id}")
        return audio_file


    def obj_to_dict(self, audio: AudioFile) -> dict:
        return {
            "id": audio.get_audio_id(),
            "name": audio.get_audio_name(),
            "last_edited": audio.get_last_edited().isoformat(),
            "duration": audio.get_duration()
        }

    def __validate_file_name(self, filename: str):
        if len(filename) == 0:
            return False
        pattern = r'^[a-zA-Z0-9\-_&$@]+$'
        if re.match(pattern, filename):
            return True
        else:
            return False