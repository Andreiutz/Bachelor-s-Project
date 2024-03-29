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

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        folder_name = preprocessService.preprocess_audio(file)
        return JSONResponse(status_code=200, content={"message": f"File uploaded successfully in folder {folder_name}"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
        folder_name = preprocessService.preprocess_audio(file)
        prediction = predictionService.predict_strums(folder_name)
        return JSONResponse(status_code=200, content=prediction)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
