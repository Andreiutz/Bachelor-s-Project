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


    def persist_audio_file(self, file : UploadFile, user_id : str):
        if not file.filename.endswith('.wav'):
            raise InvalidFileFormatException('Invalid file format')

        unique_id = str(uuid.uuid4())
        unique_file_name = f"{unique_id}.wav"
        original_file_name = file.filename.replace('.wav', '')
        date_posted = datetime.now()
        audio_file = AudioFile(audio_id=unique_id, audio_name=original_file_name, user_id=user_id, date_posted = date_posted)


        with open(os.path.join("../data/audio/wav_files/", unique_file_name), 'wb') as out_file:
            out_file.write(file.file.read())

        self.__repository.create_audio_file(audio_file=audio_file)

        return audio_file