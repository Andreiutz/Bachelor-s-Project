import sys
import tensorflow as tf
import os
import datetime
import pandas as pd
import random
import json
from DataSequence import DataSequence
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from metrics import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
os.chdir(project_root)

class NeuralNetwork:

    def __init__(self,
                 batch_size=128,
                 epochs=6,
                 frame_size=9,
                 spanning_octaves=8,
                 bins_per_octaves=36,
                 initial_learning_rate = 0.1,
                 decay_steps = 3000,
                 decay_rate = 0.3,
                 staircase = True,
                 data_path="data/archived/",
                 id_file="id_22050.csv",
                 save_path="model/training/saved/",
                 info=""):
        self.batch_size = batch_size
        self.epochs = epochs
        self.frame_size = frame_size
        self.spanning_octaves=spanning_octaves
        self.bins_per_octave = bins_per_octaves
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path
        self.more_info = info
        self.input_shape = (self.bins_per_octave * self.spanning_octaves, self.frame_size, 1)
        self.list_IDs = self.load_IDs()
        self.num_classes = 21
        self.num_strings = 6
        self.save_file_split = False

        self.partition = {}
        self.test_files = []
        self.train_files = []
        self.testing_index = -1

        self.metrics = {}


        self.save_folder = self.save_path + datetime.datetime.now().strftime("%Y-%m-%d") + "/"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)


    def load_IDs(self):
        csv_file = self.data_path + self.id_file
        return list(pd.read_csv(csv_file, header=None)[0])

    def split_files(self, train_percent):
        train = []
        test = []
        for file in os.listdir("data/audio/GuitarSet/annotation/"):
            file_name = file[:-5]
            r = random.randint(0, 100)
            if r < train_percent:
                train.append(file_name)
            else:
                test.append(file_name)
        return train, test

    def split_data_by_index(self, testing_index):
        train = []
        test = []
        for id in self.list_IDs:
            guitarist_idx = int(id.split('_')[0])
            if guitarist_idx == testing_index:
                test.append(id)
            else:
                train.append(id)

        return train, test

    def split_data_by_file_percent(self, train_percent):
        self.train_files, self.test_files = self.split_files(train_percent)
        train = []
        test  =[]
        for id in self.list_IDs:
            file_id = '_'.join(id.split('_')[:-1])
            if file_id in self.train_files:
                train.append(id)
            else:
                test.append(id)
        return train, test

    def split_data(self, testing_index=-1, file_train_percent=-1, folder_name = ""):
        if 0 <= testing_index < 6:
            self.testing_index = testing_index
            self.partition["train"], self.partition["test"] = self.split_data_by_index(testing_index)
            folder_name = self.testing_index
        elif 1 <= file_train_percent <= 99:
            self.save_file_split = True
            self.partition["train"], self.partition["test"] = self.split_data_by_file_percent(file_train_percent)
            if folder_name == "":
                raise "Invalid save folder name"
        else:
            raise "Invalid split method"

        self.current_training_folder = self.save_folder + str(folder_name) + "_" + str(self.__get_number_of_attempts(self.save_folder, folder_name) + 1) + "_" + str(self.spanning_octaves) + "_octaves" + "/"
        if not os.path.exists(self.current_training_folder):
            os.makedirs(self.current_training_folder)

        self.training_sequence = DataSequence(self.partition["train"],
                                              batch_size=self.batch_size,
                                              data_path=f"data/archived/GuitarSet/{self.spanning_octaves}_octaves/",
                                              shuffle=True,
                                              frame_size=self.frame_size)

        self.validation_sequence = DataSequence(self.partition["test"],
                                                batch_size=400,
                                                data_path=f"data/archived/GuitarSet/{self.spanning_octaves}_octaves/",
                                                shuffle=False,
                                                frame_size=self.frame_size)


    def build_model(self):
        input_layer = Input(self.input_shape)
        conv2d_1 = Conv2D(32, (3,3), activation='relu')(input_layer)
        conv2d_2 = Conv2D(64, (3,3), activation='relu')(conv2d_1)
        conv2d_3 = Conv2D(64, (3,3), activation='relu')(conv2d_2)
        max_pooling_3 = MaxPooling2D(pool_size=(2,2))(conv2d_3)
        dropout_1 = Dropout(0.6)(max_pooling_3)
        flatten_1 = Flatten()(dropout_1)

        # EString output
        E_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        E_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(E_dense_1)
        E_dropout_1 = Dropout(0.5)(E_dense_2)
        E_dense_3 = Dense(63, activation='relu')(E_dropout_1)
        EString_output = Dense(self.num_classes, activation='softmax', name='EString')(E_dense_3)

        # AString output
        A_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        A_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(A_dense_1)
        A_dropout_1 = Dropout(0.5)(A_dense_2)
        A_dense_3 = Dense(63, activation='relu')(A_dropout_1)
        AString_output = Dense(self.num_classes, activation='softmax', name='AString')(A_dense_3)


        # DString output
        D_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        D_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(D_dense_1)
        D_dropout_1 = Dropout(0.5)(D_dense_2)
        D_dense_3 = Dense(63, activation='relu')(D_dropout_1)
        DString_output = Dense(self.num_classes, activation='softmax', name='DString')(D_dense_3)


        # GString output
        G_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        G_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(G_dense_1)
        G_dropout_1 = Dropout(0.5)(G_dense_2)
        G_dense_3 = Dense(63, activation='relu')(G_dropout_1)
        GString_output = Dense(self.num_classes, activation='softmax', name='GString')(G_dense_3)


        # BString output
        B_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        B_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(B_dense_1)
        B_dropout_1 = Dropout(0.5)(B_dense_2)
        B_dense_3 = Dense(63, activation='relu')(B_dropout_1)
        BString_output = Dense(self.num_classes, activation='softmax', name='BString')(B_dense_3)


        # eString output
        e_dense_1 = Dense(252, activation='relu', kernel_regularizer=l2(0.003))(flatten_1)
        e_dense_2 = Dense(126, activation='relu', kernel_regularizer=l2(0.003))(e_dense_1)
        e_dropout_1 = Dropout(0.5)(e_dense_2)
        e_dense_3 = Dense(63, activation='relu')(e_dropout_1)
        eString_output = Dense(self.num_classes, activation='softmax', name='eString')(e_dense_3)


        # Creating the model
        model = Model(inputs=input_layer,
                      outputs=[EString_output, AString_output, DString_output, GString_output, BString_output,
                               eString_output])


        lr_schedule = ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase)

        sgd_optimizer = SGD(learning_rate=lr_schedule)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd_optimizer,
                      metrics=['accuracy'])

        self.model = model

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
                    self.training_sequence,
                    epochs=self.epochs,
                    use_multiprocessing=True,
                    workers=6
                )
            else:
                checkpoint_path = self.current_training_folder + "checkpoints"
                os.makedirs(checkpoint_path)
                callback = ModelCheckpoint(filepath=checkpoint_path + "/weights_epoch_{epoch:02d}.h5", save_weights_only=True, verbose=1)
                self.model.fit(
                    self.training_sequence,
                    epochs=self.epochs,
                    use_multiprocessing=True,
                    workers=6,
                    callbacks=[callback]
                )

    def test(self):
        self.y_gt = np.empty((len(self.partition["test"]), 6, self.num_classes))
        self.y_pred = np.empty((len(self.partition["test"]), 6, self.num_classes))
        index = 0
        for i in range(len(self.validation_sequence)):
            X_test, y_gt = self.validation_sequence[i]
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

    def evaluate(self):
        self.metrics["pitch_precision"] = pitch_precision(self.y_pred, self.y_gt)
        self.metrics["pitch_recall"] = pitch_recall(self.y_pred, self.y_gt)
        self.metrics["pitch_f_score"] = pitch_f_score(self.y_pred, self.y_gt)
        self.metrics["tab_precision"] = tab_precision(self.y_pred, self.y_gt)
        self.metrics["tab_recall"] = tab_recall(self.y_pred, self.y_gt)
        self.metrics["tab_f_score"] = tab_f_measure(self.y_pred, self.y_gt)

    def plot_model(self, file_name):
        plot_model(self.model, to_file=f"{self.current_training_folder}{file_name}",
                   expand_nested=True, show_shapes=True)

    def save_model(self):
        self.model.save(self.current_training_folder + "model")

    def save_predictions(self):
        np.savez(self.current_training_folder + "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)

    def save_results_csv(self):
        with open(self.current_training_folder + 'metrics.json', 'w') as file:
            json.dump(self.metrics, file, indent=4)

    def log_model_details(self):
        log_file = self.current_training_folder + "log.txt"
        with open(log_file, 'w') as log:
            log.write("\ntest index " + str(self.testing_index))
            log.write("\nbatch_size: " + str(self.batch_size))
            log.write("\nepochs: " + str(self.epochs))
            log.write("\nbins per octave: " + str(self.bins_per_octave))
            log.write("\ndata_path: " + str(self.data_path))
            log.write("\ncon_win_size: " + str(self.frame_size))
            log.write("\nid_file: " + str(self.id_file))
            log.write("\ninitial learning rate: " + str(self.initial_learning_rate))
            log.write("\ndecay steps: " + str(self.decay_steps))
            log.write("\ndecay rate: " + str(self.decay_rate))
            log.write("\nstaircase: " + str(self.staircase))
            log.write("\nother info: " + self.more_info + "\n")
        if self.save_file_split:
            train_file_name = self.current_training_folder + "train_files.txt"
            test_file_name = self.current_training_folder + "test_files.txt"
            with open(train_file_name, 'w') as f:
                for train_file in self.train_files:
                    f.write(f"{train_file}\n")
            with open(test_file_name, 'w') as f:
                for test_file in self.test_files:
                    f.write(f"{test_file}\n")

    def __get_number_of_attempts(self, folder_path, fold_index):
        if not os.path.isdir(folder_path):
            raise ValueError(f"The provided folder path '{folder_path}' is not a valid directory.")

        count = 0
        for folder in os.listdir(folder_path):
            if folder.startswith(str(fold_index)) and os.path.isdir(os.path.join(folder_path, folder)):
                count += 1

        return count