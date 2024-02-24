from server.service.PreprocessService import PreprocessService
from server.service.PredictionService import PredictionService

def test_prediction():
    preprocess_service = PreprocessService()
    prediction_service = PredictionService(preprocess_service)
    result = prediction_service.predict('B_multiple')
    print()

if __name__ == '__main__':
    test_prediction()