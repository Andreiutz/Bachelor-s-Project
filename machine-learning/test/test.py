from server.service.PreprocessService import PreprocessService
from server.service.PredictionService import PredictionService

def test_prediction():
    preprocess_service = PreprocessService()
    prediction_service = PredictionService(preprocess_service)
    prediction_service.predict('02_SS1-68-E_comp_mic')

if __name__ == '__main__':
    test_prediction()