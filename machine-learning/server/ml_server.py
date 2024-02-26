from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from server.repository.AudioFileRepository import AudioFileRepository
from server.service.AudioFileService import AudioFileService
import os
import shutil
import aiofiles

#todo
# tune onset detection
# improve model
# improve strum by batch (ignore silence)

from server.service.PredictionService import PredictionService
from server.service.PreprocessService import PreprocessService

app = FastAPI()

UPLOAD_FOLDER = '../data/audio/wav_files'

repository = AudioFileRepository(
    host='localhost',
    dbname='guitar_ml',
    password='postgres',
    user='postgres'
)

audioFileService = AudioFileService(repository=repository)
preprocessService = PreprocessService()
predictionService = PredictionService(preprocess_service=preprocessService)

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        audio_file=audioFileService.persist_audio_file(file=file, user_id='test')
        return JSONResponse(status_code=200, content={"message": "File uploaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_audio(file: UploadFile = File(...)):
    try:
        folder_name = preprocessService.preprocess_audio(file)
        prediction = predictionService.predict_strums(folder_name)
        return JSONResponse(status_code=200, content=prediction)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
