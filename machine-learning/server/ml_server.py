import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
os.chdir(project_root)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from server.repository.AudioFileRepository import AudioFileRepository
from server.service.AudioFileService import AudioFileService


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


@app.get("/predict")
def predict_cached_audio(name: str):
    try:
        prediction = predictionService.predict_strums(name)
        return JSONResponse(status_code=200, content=prediction)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))



@app.post("/predict")
def predict_audio(file: UploadFile = File(...)):
    try:
        audio_file = audioFileService.persist_audio_file(file)
        preprocessService.preprocess_audio(file, audio_file.get_audio_id())
        prediction = predictionService.predict_all(audio_file.get_audio_id())
        return JSONResponse(status_code=200, content=prediction)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

@app.get("/songs")
def get_audio_list():
    try:
        audio_files = repository.get_all_audio_files()
        return JSONResponse(status_code=200, content=[
            {
                "id" : a.get_audio_id(),
                "name" : a.get_audio_name(),
                "last_edited" : a.get_last_edited().isoformat()
            }
            for a in audio_files
        ])
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
