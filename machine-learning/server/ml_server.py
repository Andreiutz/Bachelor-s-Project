import sys
import os

from starlette.middleware.cors import CORSMiddleware

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
os.chdir(project_root)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from fastapi.responses import JSONResponse
from server.repository.AudioFileRepository import AudioFileRepository
from server.service.AudioFileService import AudioFileService
from server.service.PredictionService import PredictionService
from server.service.PreprocessService import PreprocessService
from server.service.CacheService import CacheService
from server.exceptions.FileNotFoundException import FileNotFoundException

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
repository = AudioFileRepository(
    host='localhost',
    dbname='guitar_ml',
    password='postgres',
    user='postgres'
)

audioFileService = AudioFileService(repository=repository)
preprocessService = PreprocessService()
cache_service = CacheService()
predictionService = PredictionService(preprocess_service=preprocessService, cache_service=cache_service)

@app.get("/file")
def get_audio_file(audio_id: str):
    try:
        file_path = audioFileService.get_audio_path(audio_id)
        return FileResponse(file_path, media_type="audio/wav")
    except FileNotFoundException:
        return HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audio file: {str(e)}")

@app.get("/predict-tab")
def predict_tablature_from_folder(name: str, load: bool):
    try:
        prediction = predictionService.predict_tablature(name, load=load)
        return JSONResponse(status_code=200, content=prediction)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

@app.post("/predict-tab")
def predict_tablature_from_upload(file: UploadFile = File(...)):
    try:
        audio_file = audioFileService.persist_audio_file(file)
        preprocessService.archive_file_from_upload(file, audio_file.get_audio_id())
        prediction = predictionService.predict_tablature(audio_file.get_audio_id(), cache=True)
        return JSONResponse(status_code=200, content=prediction)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))


@app.get("/predict-full")
def predict_full_from_folder(name: str, load: bool):
    try:
        prediction = predictionService.predict_full_samples(name, load=load)
        return JSONResponse(status_code=200, content=prediction)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        audio_file = audioFileService.persist_audio_file(file)
        preprocessService.archive_file_from_upload(file, audio_file.get_audio_id())
        return JSONResponse(status_code=200, content=audioFileService.obj_to_dict(audio_file))
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

@app.post("/predict-full")
def predict_full_from_upload(file: UploadFile = File(...)):
    try:
        audio_file = audioFileService.persist_audio_file(file)
        preprocessService.archive_file_from_upload(file, audio_file.get_audio_id())
        prediction = predictionService.predict_full_samples(audio_file.get_audio_id(), cache=True)
        return JSONResponse(status_code=200, content=prediction)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))


@app.get("/songs")
def get_audio_list():
    try:
        audio_files = repository.get_all()
        return JSONResponse(status_code=200, content=[
            audioFileService.obj_to_dict(a)
            for a in audio_files
        ])
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

@app.delete("/songs")
def delete_audio(audio_id: str):
    try:
        audio_file = audioFileService.delete_audio(audio_id)
        return JSONResponse(status_code=200, content=[
            audioFileService.obj_to_dict(audio_file)
        ])
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
