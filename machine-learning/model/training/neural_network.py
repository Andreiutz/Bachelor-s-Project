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
                 epochs=6,
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
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []
        self.metrics["data"] = ["g0", "g1", "g2", "g3", "g4", "g5", "mean", "std dev"]

        self.bins_per_octave = 36

        self.input_shape = (self.bins_per_octave * 8, self.con_win_size, 1)

        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6

        #Values for optimizer
        self.initial_learning_rate = 0.01  # Set your initial learning rate
        self.decay_steps = 4500
        self.decay_rate = 0.66
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
        self.partition["training"] = []
        self.partition["validation"] = []
        if data_split >= 0:
            for ID in self.list_IDs:
                guitarist = int(ID.split("_")[0])
                if guitarist == data_split:
                    self.partition["validation"].append(ID)
                else:
                    self.partition["training"].append(ID)
        else:
            if partition:
                for ID in self.list_IDs:
                    chance = random.randint(0, 10)
                    if chance < 1:
                        self.partition["validation"].append(ID)
                    else:
                        self.partition["training"].append(ID)
            else:
                for ID in self.list_IDs:
                    self.partition["training"].append(ID)

        self.training_generator = DataGenerator(self.partition['training'],
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                con_win_size=self.con_win_size)

        self.validation_generator = DataGenerator(self.partition['validation'],
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

        dropout_1 = Dropout(0.5)(max_pooling_3)

        flatten_1 = Flatten()(dropout_1)

        # EString output
        E_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        E_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(E_dense_1)
        E_dropout_1 = Dropout(0.5)(E_dense_2)
        E_dense_3 = Dense(63, activation='relu')(E_dropout_1)
        EString_output = Dense(21, activation='softmax', name='EString')(E_dense_3)

        # AString output
        A_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        A_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(A_dense_1)
        A_dropout_1 = Dropout(0.5)(A_dense_2)
        A_dense_3 = Dense(63, activation='relu')(A_dropout_1)
        AString_output = Dense(21, activation='softmax', name='AString')(A_dense_3)


        # DString output
        D_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        D_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(D_dense_1)
        D_dropout_1 = Dropout(0.5)(D_dense_2)
        D_dense_3 = Dense(63, activation='relu')(D_dropout_1)
        DString_output = Dense(21, activation='softmax', name='DString')(D_dense_3)


        # GString output
        G_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        G_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(G_dense_1)
        G_dropout_1 = Dropout(0.5)(G_dense_2)
        G_dense_3 = Dense(63, activation='relu')(G_dropout_1)
        GString_output = Dense(21, activation='softmax', name='GString')(G_dense_3)


        # BString output
        B_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        B_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(B_dense_1)
        B_dropout_1 = Dropout(0.5)(B_dense_2)
        B_dense_3 = Dense(63, activation='relu')(B_dropout_1)
        BString_output = Dense(21, activation='softmax', name='BString')(B_dense_3)


        # eString output
        e_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        e_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(e_dense_1)
        e_dropout_1 = Dropout(0.5)(e_dense_2)
        e_dense_3 = Dense(63, activation='relu')(e_dropout_1)
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
        # adam_optimizer = Adam(learning_rate=lr_schedule)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd_optimizer,
                      metrics=['accuracy'])

        self.model = model

    def plot_model(self, file_name):
        plot_model(self.model, to_file=f"{self.split_folder}{file_name}",
                                  expand_nested=True, show_shapes=True)

    def train(self, load=False):
        if load:
            #self.model = tf.keras.models.load_model("saved/c_3bin_2024-02-20/2_2/model")
            pass
        else:
            self.model.fit(
                self.training_generator,
                epochs=self.epochs,
                use_multiprocessing=True,
                workers=6
            )

    def save_model(self):
        #self.model.save_weights(self.split_folder + "weights.h5")
        self.model.save(self.split_folder + "model")

    def test(self):
        self.y_gt = np.empty((len(self.partition["validation"]), 6, 21))
        self.y_pred = np.empty((len(self.partition["validation"]), 6, 21))
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
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))

    def save_results_csv(self):
        output = {}
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]
        # output["data"] =  self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.split_folder + "results.csv")

if __name__ == '__main__':
    configure_gpu()
    tabcnn = NeuralNetwork(info="L1=0.003 for first 2 dense layers of each string")
    for fold in range(2,3):
        print("\nfold " + str(fold))
        tabcnn.partition_data(data_split=fold)
        print("building model...")
        tabcnn.build_model()
        print("logging model...")
        tabcnn.log_model()
        tabcnn.plot_model('model_architecture.png')
        print("training...")
        tabcnn.train()
        print("saving weights...")
        tabcnn.save_model()
        print("testing...")
        tabcnn.test()
        tabcnn.save_predictions()
        print("evaluation...")
        tabcnn.evaluate()
        print("saving results...")
        tabcnn.save_results_csv()
