from server.service.PreprocessService import PreprocessService
from server.service.PredictionService import PredictionService

def test_prediction():
    preprocess_service = PreprocessService()
    prediction_service = PredictionService(preprocess_service)

    f = 'b65db1b1-c16c-457f-aad7-a1b22fbd4d36'

    preprocess_service.archive_file_from_folder(f)
    result = prediction_service.predict_strums(f)



    print()

if __name__ == '__main__':
    test_prediction()