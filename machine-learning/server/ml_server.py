from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from server.repository.AudioFileRepository import AudioFileRepository
from server.service.AudioFileService import AudioFileService
import os
import shutil
import aiofiles

app = FastAPI()

UPLOAD_FOLDER = '../data/audio/wav_files'

repository = AudioFileRepository(
    host='localhost',
    dbname='guitar_ml',
    password='postgres',
    user='postgres'
)

service = AudioFileService(repository=repository)

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        audio_file=service.persist_audio_file(file=file, user_id='test')
        return JSONResponse(status_code=200, content={"message": "File uploaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
