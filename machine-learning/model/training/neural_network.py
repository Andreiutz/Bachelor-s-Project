import sys

import tensorflow as tf
import os
import datetime
import pandas as pd
import random
import numpy as np
from DataGenerator import DataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation, Input
from keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda
from keras import backend as K
from metrics import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from keras.optimizers import SGD
# from keras.optimizers.schedules import ExponentialDecay
from keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
os.chdir(project_root)
print(project_root)

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

class NeuralNetwork:

    def __init__(self,
                 batch_size=128,
                 epochs=6,
                 con_win_size=9,
                 spanning_octaves=8,
                 bins_per_octaves=36,
                 data_path="data/archived/",
                 id_file="id_22050.csv",
                 save_path="model/training/saved/",
                 info=""):
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spanning_octaves=spanning_octaves
        self.bins_per_octave = bins_per_octaves
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path

        self.load_IDs()

        self.save_folder = self.save_path + datetime.datetime.now().strftime("%Y-%m-%d") + "/"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.metrics = {}
        self.metrics["pitch_precision"] = []
        self.metrics["pitch_recall"] = []
        self.metrics["pitch_f_score"] = []
        self.metrics["tab_precision"] = []
        self.metrics["tab_recall"] = []
        self.metrics["tab_f_score"] = []


        self.input_shape = (self.bins_per_octave * self.spanning_octaves, self.con_win_size, 1)

        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6

        #Values for optimizer
        self.initial_learning_rate = 0.05  # Set your initial learning rate
        self.decay_steps = 3500
        self.decay_rate = 0.6
        self.use_momentum=False
        self.momentum=0.9

        self.staircase = True

        self.more_info = info

    def load_IDs(self):
        csv_file = self.data_path + self.id_file
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])

    def __get_number_of_attempts(self, folder_path, fold_index):
        if not os.path.isdir(folder_path):
            raise ValueError(f"The provided folder path '{folder_path}' is not a valid directory.")

        count = 0
        for folder in os.listdir(folder_path):
            if folder.startswith(str(fold_index)) and os.path.isdir(os.path.join(folder_path, folder)):
                count += 1

        return count

    def partition_data(self, partition=True, training_index=-1, folder_name=""):
        if training_index >= 0:
            self.training_index = training_index
        else:
            self.training_index = folder_name
        self.partition = {}
        self.partition["train"] = []
        self.partition["test"] = []
        if training_index >= 0:
            for ID in self.list_IDs:
                guitarist = int(ID.split("_")[0])
                if guitarist == training_index:
                    self.partition["test"].append(ID)
                else:
                    self.partition["train"].append(ID)
        else:
            if partition:
                for ID in self.list_IDs:
                    chance = random.randint(0, 10)
                    if chance < 1:
                        self.partition["test"].append(ID)
                    else:
                        self.partition["train"].append(ID)
            else:
                for ID in self.list_IDs:
                    self.partition["train"].append(ID)

        self.training_generator = DataGenerator(self.partition["train"],
                                                batch_size=self.batch_size,
                                                data_path=f"data/archived/GuitarSet/{self.spanning_octaves}_octaves/",
                                                shuffle=True,
                                                con_win_size=self.con_win_size)

        self.validation_generator = DataGenerator(self.partition["test"],
                                                  batch_size=400, #to modify
                                                  data_path=f"data/archived/GuitarSet/{self.spanning_octaves}_octaves/",
                                                  shuffle=False,
                                                  con_win_size=self.con_win_size)

        self.current_training_folder = self.save_folder + str(self.training_index) + "_" + str(self.__get_number_of_attempts(self.save_folder, self.training_index) + 1) + "_" + str(self.spanning_octaves) + "_octaves" + "/"
        if not os.path.exists(self.current_training_folder):
            os.makedirs(self.current_training_folder)


    def log_model_details(self):
        self.log_file = self.current_training_folder + "log.txt"
        with open(self.log_file, 'w') as log:
            log.write("\nbatch_size: " + str(self.batch_size))
            log.write("\nepochs: " + str(self.epochs))
            log.write("\nbins per octave: " + str(self.bins_per_octave))
            log.write("\nuse momentum: " + str(self.use_momentum))
            log.write("\nmomentum value: " + str(self.momentum))
            log.write("\ndata_path: " + str(self.data_path))
            log.write("\ncon_win_size: " + str(self.con_win_size))
            log.write("\nid_file: " + str(self.id_file))
            log.write("\ninitial learning rate: " + str(self.initial_learning_rate))
            log.write("\ndecay steps: " + str(self.decay_steps))
            log.write("\ndecay rate: " + str(self.decay_rate))
            log.write("\nstaircase: " + str(self.staircase))
            log.write("\nother info: " + self.more_info + "\n")

    def build_model(self):
        input_layer = Input(self.input_shape)

        conv2d_1 = Conv2D(32, (3,3), activation='relu')(input_layer)

        conv2d_2 = Conv2D(64, (3,3), activation='relu')(conv2d_1)

        conv2d_3 = Conv2D(64, (3,3), activation='relu')(conv2d_2)

        max_pooling_3 = MaxPooling2D(pool_size=(2,2))(conv2d_3)

        dropout_1 = Dropout(0.4)(max_pooling_3)

        flatten_1 = Flatten()(dropout_1)

        # EString output
        E_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        E_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.001))(E_dense_1)
        E_dropout_1 = Dropout(0.5)(E_dense_2)
        E_dense_3 = Dense(42, activation='relu')(E_dropout_1)
        EString_output = Dense(21, activation='softmax', name='EString')(E_dense_3)

        # AString output
        A_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        A_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.001))(A_dense_1)
        A_dropout_1 = Dropout(0.4)(A_dense_2)
        A_dense_3 = Dense(42, activation='relu')(A_dropout_1)
        AString_output = Dense(21, activation='softmax', name='AString')(A_dense_3)


        # DString output
        D_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        D_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.001))(D_dense_1)
        D_dropout_1 = Dropout(0.4)(D_dense_2)
        D_dense_3 = Dense(42, activation='relu')(D_dropout_1)
        DString_output = Dense(21, activation='softmax', name='DString')(D_dense_3)


        # GString output
        G_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        G_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.001))(G_dense_1)
        G_dropout_1 = Dropout(0.4)(G_dense_2)
        G_dense_3 = Dense(42, activation='relu')(G_dropout_1)
        GString_output = Dense(21, activation='softmax', name='GString')(G_dense_3)


        # BString output
        B_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        B_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.001))(B_dense_1)
        B_dropout_1 = Dropout(0.4)(B_dense_2)
        B_dense_3 = Dense(42, activation='relu')(B_dropout_1)
        BString_output = Dense(21, activation='softmax', name='BString')(B_dense_3)


        # eString output
        e_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        e_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.001))(e_dense_1)
        e_dropout_1 = Dropout(0.4)(e_dense_2)
        e_dense_3 = Dense(42, activation='relu')(e_dropout_1)
        eString_output = Dense(21, activation='softmax', name='eString')(e_dense_3)


        # Creating the model
        model = Model(inputs=input_layer,
                      outputs=[EString_output, AString_output, DString_output, GString_output, BString_output,
                               eString_output])


        lr_schedule = ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase)

        sgd_optimizer = SGD(learning_rate=lr_schedule) #momentum=0.6
        if self.use_momentum:
            sgd_optimizer = SGD(learning_rate=lr_schedule, momentum=self.momentum)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd_optimizer,
                      metrics=['accuracy'])

        self.model = model

    def plot_model(self, file_name):
        plot_model(self.model, to_file=f"{self.current_training_folder}{file_name}",
                   expand_nested=True, show_shapes=True)

    def train(self, load=False, model_path = "", epoch = -1, checkpoints=False):
        if load:
            self.model = tf.keras.models.load_model(model_path)
            self.more_info += f"\nmodel loaded from {model_path}"
            if epoch > 0:
                self.model.load_weights(model_path + f"/../checkpoints/weights_epoch_0{epoch}.h5")
                self.more_info += f"\nepoch {epoch}"
        else:
            if not checkpoints:
                self.model.fit(
                    self.training_generator,
                    epochs=self.epochs,
                    use_multiprocessing=True,
                    workers=6
                )
            else:
                checkpoint_path = self.current_training_folder + "checkpoints"
                os.makedirs(checkpoint_path)
                callback = ModelCheckpoint(filepath=checkpoint_path + "/weights_epoch_{epoch:02d}.h5", save_weights_only=True, verbose=1)
                self.model.fit(
                    self.training_generator,
                    epochs=self.epochs,
                    use_multiprocessing=True,
                    workers=6,
                    callbacks=[callback]
                )

    def save_model(self):
        self.model.save(self.current_training_folder + "model")

    def test(self):
        self.y_gt = np.empty((len(self.partition["test"]), 6, 21))
        self.y_pred = np.empty((len(self.partition["test"]), 6, 21))
        index = 0
        for i in range(len(self.validation_generator)):
            X_test, y_gt = self.validation_generator[i]
            y_pred = self.model.predict(X_test, verbose=0)
            size = len(y_pred[0])
            for sample_index in range(size):
                EString_pred = y_pred[0][sample_index]
                AString_pred = y_pred[1][sample_index]
                DString_pred = y_pred[2][sample_index]
                GString_pred = y_pred[3][sample_index]
                BString_pred = y_pred[4][sample_index]
                eString_pred = y_pred[5][sample_index]
                sample_tab_pred = np.array([EString_pred, AString_pred, DString_pred, GString_pred, BString_pred, eString_pred])
                self.y_pred[index,] = sample_tab_pred

                EString_gt = y_gt["EString"][sample_index]
                AString_gt = y_gt["AString"][sample_index]
                DString_gt = y_gt["DString"][sample_index]
                GString_gt = y_gt["GString"][sample_index]
                BString_gt = y_gt["BString"][sample_index]
                eString_gt = y_gt["eString"][sample_index]
                sample_tab_gt = np.array([EString_gt, AString_gt, DString_gt, GString_gt, BString_gt, eString_gt])
                self.y_gt[index,] = sample_tab_gt
                index += 1

    def save_predictions(self):
        np.savez(self.current_training_folder + "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)

    def evaluate(self):
        self.metrics["pitch_precision"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pitch_recall"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pitch_f_score"].append(pitch_f_score(self.y_pred, self.y_gt))
        self.metrics["tab_precision"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tab_recall"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tab_f_score"].append(tab_f_measure(self.y_pred, self.y_gt))

    def save_results_csv(self):
        df = pd.DataFrame.from_dict(self.metrics)
        df.to_csv(self.current_training_folder + "results.csv")

if __name__ == '__main__':
    configure_gpu()
    neural_network = NeuralNetwork(info="L1=0.003 for first layer, 0.001 for second (dense layers for ech string)\n0.4 dropout everywhere", spanning_octaves=8)
    test_index = 2
    print("\ntest index " + str(test_index))
    neural_network.partition_data(training_index=test_index)

    print("building model...")
    neural_network.build_model()

    print("plotting model...")
    neural_network.plot_model('model_architecture.png')

    print("training...")
    neural_network.train(checkpoints=True, load=False, model_path="model/training/saved/2024-03-13/2_1_5_octaves/model", epoch=6)

    print("saving weights...")
    neural_network.save_model()

    print("testing...")
    neural_network.test()

    print("saving predictions...")
    neural_network.save_predictions()

    print("evaluation...")
    neural_network.evaluate()

    print("saving results...")
    neural_network.save_results_csv()

    print("logging model...")
    neural_network.log_model_details()