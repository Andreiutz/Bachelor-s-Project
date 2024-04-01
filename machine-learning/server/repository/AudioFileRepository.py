import psycopg2

from server.domain.AudioFIle import AudioFile


class AudioFileRepository:
    def __init__(self, host, dbname, user, password, port=5432):
        self.conn_params = {
            "host": host,
            "dbname": dbname,
            "user": user,
            "password": password,
            "port": port
        }

    def connect(self):
        return psycopg2.connect(**self.conn_params)

    def create_audio_file(self, audio_file: AudioFile):
        query = """
        INSERT INTO audio_files (audio_id, audio_name, last_edited) 
        VALUES (%s, %s, %s)
        """
        values = (audio_file.get_audio_id(), audio_file.get_audio_name(),
                   audio_file.get_last_edited())

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, values)
                conn.commit()

    def get_audio_file(self, audio_id):
        query = "SELECT * FROM audio_files WHERE audio_id = %s"
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (audio_id,))
                result = cur.fetchone()
                if result:
                    return AudioFile(*result)
                return None

    def get_all_audio_files(self):
        query = "SELECT * FROM audio_files"
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                results = cur.fetchall()
                audio_files = []
                for result in results:
                    audio_files.append(AudioFile(*result))
                return audio_files

