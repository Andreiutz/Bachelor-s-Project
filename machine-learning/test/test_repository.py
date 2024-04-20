import datetime
import unittest
from server.repository.AudioFileRepository import AudioFileRepository
from server.domain.AudioFile import AudioFile

class RepositoryTestClass(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(RepositoryTestClass, self).__init__(*args, **kwargs)
        # Instantiate helper objects in the constructor
        self.repository = AudioFileRepository(
            host='localhost',
            dbname='guitar_ml_test',
            password='postgres',
            user='postgres'
        )

    def setUp(self) -> None:

        audio_file_1 = AudioFile("id1", "name1", datetime.datetime.now(), 1.2)
        audio_file_2 = AudioFile("id2", "name2", datetime.datetime.now(), 1.2)
        audio_file_3 = AudioFile("id3", "name3", datetime.datetime.now(), 1.2)
        audio_file_4 = AudioFile("id4", "name4", datetime.datetime.now(), 1.2)
        self.repository.add_file(audio_file_1)
        self.repository.add_file(audio_file_2)
        self.repository.add_file(audio_file_3)
        self.repository.add_file(audio_file_4)

    def tearDown(self) -> None:
        files = self.repository.get_all()
        for file in files:
            self.repository.delete_file(file.get_audio_id())

    def test_add(self):
        files = self.repository.get_all()
        self.assertEquals(len(files), 4)
        audio_file_5 = AudioFile("id5", "name5", datetime.datetime.now(), 1.2)
        self.repository.add_file(audio_file_5)
        files = self.repository.get_all()
        self.assertEquals(len(files), 5)

    def test_delete(self):
        files = self.repository.get_all()
        self.assertEquals(len(files), 4)
        self.repository.delete_file("id2")
        self.repository.delete_file("id3")
        self.repository.delete_file("id5")
        files = self.repository.get_all()
        self.assertEquals(len(files), 2)

    def test_get_all(self):
        files = self.repository.get_all()
        self.assertEquals(len(files), 4)

    def test_get_one(self):
        audio_file = self.repository.get_file("id1")
        self.assertEquals(audio_file.get_audio_name(), "name1")
        audio_file = self.repository.get_file("id2")
        self.assertEquals(audio_file.get_audio_name(), "name2")
        audio_file = self.repository.get_file("id5")
        self.assertEquals(audio_file, None)
