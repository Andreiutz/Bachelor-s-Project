import os.path
from datetime import datetime
from server.domain.AudioFIle import AudioFile
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
        audio_file = AudioFile(audio_id=unique_id, audio_name=original_file_name, last_edited= date_posted)

        os.mkdir(f"../data/audio/{unique_id}/")

        with open(f"../data/audio/{unique_id}/audio.wav", 'wb') as out_file:
            out_file.write(file.file.read())

        self.__repository.create_audio_file(audio_file=audio_file)

        return audio_file