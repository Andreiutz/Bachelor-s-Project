import os
import sys
import tensorflow as tf
import warnings
from neural_network import NeuralNetwork

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
os.chdir(project_root)

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            print(gpus)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(device=gpu, enable=True)
        except RuntimeError as e:
            print(e)
    else:
        print('no gpus')


if __name__ == '__main__':
    configure_gpu()
    neural_network = NeuralNetwork(spanning_octaves=8, info="kernel regularizer l2 003 for conv2d layers, dropout 0.5 only for branches")

    test_index = 2
    print("\ntest guitarist index: " + str(test_index))
    neural_network.split_data(testing_index=test_index)
    #neural_network.split_data(file_train_percent=83, folder_name="train_83p")

    neural_network.build_model()
    print("model built")

    neural_network.plot_model('model_architecture.png')
    print("model plot saved")

    print("training started...")
    neural_network.train(checkpoints=True)

    neural_network.save_model()
    print("weights saved")

    print("statistics...")
    neural_network.test()

    neural_network.save_predictions()
    print("saved predictions")

    neural_network.evaluate()
    print("saved metrics")

    neural_network.save_results_csv()
    print("saved results")

    neural_network.log_model_details()
    print("logged model")