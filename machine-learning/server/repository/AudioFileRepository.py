import psycopg2

from server.domain.AudioFile import AudioFile


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

    def add_file(self, audio_file: AudioFile):
        query = """
        INSERT INTO audio_files (audio_id, audio_name, last_edited, duration) 
        VALUES (%s, %s, %s, %s)
        """
        values = (audio_file.get_audio_id(), audio_file.get_audio_name(),
                   audio_file.get_last_edited(), audio_file.get_duration())

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, values)
                conn.commit()

    def delete_file(self, audio_id):
        query = "DELETE FROM audio_files WHERE audio_id = %s"
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (audio_id,))
                conn.commit()

    def get_file(self, audio_id):
        query = "SELECT * FROM audio_files WHERE audio_id = %s"
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (audio_id,))
                result = cur.fetchone()
                if result:
                    return AudioFile(*result)
                return None

    def get_all(self):
        query = "SELECT * FROM audio_files"
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                results = cur.fetchall()
                audio_files = []
                for result in results:
                    audio_files.append(AudioFile(*result))
                return audio_files

