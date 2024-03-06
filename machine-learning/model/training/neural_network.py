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
                 epochs=4,
                 con_win_size=9,
                 data_path="../../data/archived/",
                 id_file="id_22050.csv",
                 save_path="saved/",
                 info=""):
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
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

        self.bins_per_octave = 36

        self.input_shape = (self.bins_per_octave * 8, self.con_win_size, 1)

        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6

        #Values for optimizer
        self.initial_learning_rate = 0.01  # Set your initial learning rate
        self.decay_steps = 4000
        self.decay_rate = 0.7
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

    def partition_data(self, partition=True, data_split=-1, folder_name=""):
        if data_split >= 0:
            self.data_split = data_split
        else:
            self.data_split = folder_name
        self.partition = {}
        self.partition["train"] = []
        self.partition["test"] = []
        if data_split >= 0:
            for ID in self.list_IDs:
                guitarist = int(ID.split("_")[0])
                if guitarist == data_split:
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
                                                shuffle=True,
                                                con_win_size=self.con_win_size)

        self.validation_generator = DataGenerator(self.partition["test"],
                                                  batch_size=400, #to modify
                                                  shuffle=False,
                                                  con_win_size=self.con_win_size)

        self.split_folder = self.save_folder + str(self.data_split) + "_" + str(self.__get_number_of_attempts(self.save_folder, self.data_split) + 1) + "/"
        if not os.path.exists(self.split_folder):
            os.makedirs(self.split_folder)


    def log_model(self):
        self.log_file = self.split_folder + "log.txt"
        with open(self.log_file, 'w') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nbins per octave: " + str(self.bins_per_octave))
            fh.write("\nuse momentum: " + str(self.use_momentum))
            fh.write("\nmomentum value: " + str(self.momentum))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file))
            fh.write("\ninitial learning rate: " + str(self.initial_learning_rate))
            fh.write("\ndecay steps: " + str(self.decay_steps))
            fh.write("\ndecay rate: " + str(self.decay_rate))
            fh.write("\nstaircase: " + str(self.staircase))
            fh.write("\nother info: " + self.more_info + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def build_model(self):
        input_layer = Input(self.input_shape)

        conv2d_1 = Conv2D(32, (3,3), activation='relu')(input_layer)

        conv2d_2 = Conv2D(64, (3,3), activation='relu')(conv2d_1)

        conv2d_3 = Conv2D(64, (3,3), activation='relu')(conv2d_2)

        max_pooling_3 = MaxPooling2D(pool_size=(2,2))(conv2d_3)

        dropout_1 = Dropout(0.6)(max_pooling_3)

        flatten_1 = Flatten()(dropout_1)

        # EString output
        E_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        E_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(E_dense_1)
        E_dropout_1 = Dropout(0.5)(E_dense_2)
        E_dense_3 = Dense(42, activation='relu')(E_dropout_1)
        EString_output = Dense(21, activation='softmax', name='EString')(E_dense_3)

        # AString output
        A_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        A_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(A_dense_1)
        A_dropout_1 = Dropout(0.5)(A_dense_2)
        A_dense_3 = Dense(42, activation='relu')(A_dropout_1)
        AString_output = Dense(21, activation='softmax', name='AString')(A_dense_3)


        # DString output
        D_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        D_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(D_dense_1)
        D_dropout_1 = Dropout(0.5)(D_dense_2)
        D_dense_3 = Dense(42, activation='relu')(D_dropout_1)
        DString_output = Dense(21, activation='softmax', name='DString')(D_dense_3)


        # GString output
        G_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        G_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(G_dense_1)
        G_dropout_1 = Dropout(0.5)(G_dense_2)
        G_dense_3 = Dense(42, activation='relu')(G_dropout_1)
        GString_output = Dense(21, activation='softmax', name='GString')(G_dense_3)


        # BString output
        B_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        B_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(B_dense_1)
        B_dropout_1 = Dropout(0.5)(B_dense_2)
        B_dense_3 = Dense(42, activation='relu')(B_dropout_1)
        BString_output = Dense(21, activation='softmax', name='BString')(B_dense_3)


        # eString output
        e_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        e_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(e_dense_1)
        e_dropout_1 = Dropout(0.5)(e_dense_2)
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
        plot_model(self.model, to_file=f"{self.split_folder}{file_name}",
                                  expand_nested=True, show_shapes=True)

    def train(self, load=False, model_path = "", checkpoints=False):
        if load:
            self.model = tf.keras.models.load_model(model_path)
            pass
        else:
            if not checkpoints:
                self.model.fit(
                    self.training_generator,
                    epochs=self.epochs,
                    use_multiprocessing=True,
                    workers=6
                )
            else:
                checkpoint_path = self.split_folder + "checkpoints"
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
        self.model.save(self.split_folder + "model")

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
        np.savez(self.split_folder + "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)

    def evaluate(self):
        self.metrics["pitch_precision"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pitch_recall"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pitch_f_score"].append(pitch_f_score(self.y_pred, self.y_gt))
        self.metrics["tab_precision"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tab_recall"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tab_f_score"].append(tab_f_measure(self.y_pred, self.y_gt))

    def save_results_csv(self):
        df = pd.DataFrame.from_dict(self.metrics)
        df.to_csv(self.split_folder + "results.csv")

if __name__ == '__main__':
    configure_gpu()
    neural_network = NeuralNetwork(info="L1=0.003 for first 2 dense layers of each string")
    test_index = 2
    print("\ntest index " + str(test_index))
    neural_network.partition_data(data_split=test_index)

    print("building model...")
    neural_network.build_model()

    print("logging model...")
    neural_network.log_model()

    print("plotting model...")
    neural_network.plot_model('model_architecture.png')

    print("training...")
    neural_network.train(checkpoints=True)

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
