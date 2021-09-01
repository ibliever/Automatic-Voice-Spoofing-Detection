import tensorflow as tf
import network
import pickle
from tensorflow import keras
import pandas as pd
import numpy as np

train_count = 1
eval_count = 1

def preprocess(batch_x, lab):
    print("============preprocess==============")
    set = []
    if (lab == 0):
        path = "data/temp/train/"
    elif (lab == 1):
        path = "data/temp/eval/"
    else:
        path = "data/temp/dev/"
    i = 0
    for x in batch_x:
        f = open(path + x + ".pkl", 'rb')
        train_data = pickle.load(f)
        train_data = np.reshape(train_data, (1, 64, 64, 1))
        if (i==0):
            set = train_data
        else:
            set = np.concatenate((set, train_data), axis=0)

        i = i + 1
        # print("*********************")
        # print(set.shape)
    return set



if __name__ == '__main__':
    train_datasets = pd.read_csv("ASVspoof2019.LA.cm.train.trn.csv", header=None, names=["SPEAKER_ID", "AUDIO_FILE_NAME", "SYSTEM_ID", "-", "KEY"])
    eval_datasets = pd.read_csv("ASVspoof2019.LA.cm.eval.trl.csv", header=None, names=["SPEAKER_ID", "AUDIO_FILE_NAME", "SYSTEM_ID", "-", "KEY"])
    dev_datasets = pd.read_csv("ASVspoof2019.LA.cm.dev.trl.csv", header=None, names=["SPEAKER_ID", "AUDIO_FILE_NAME", "SYSTEM_ID", "-", "KEY"])
    train_file = train_datasets.AUDIO_FILE_NAME
    train_label = train_datasets.KEY
    eval_file = eval_datasets.AUDIO_FILE_NAME
    eval_label = eval_datasets.KEY
    dev_file = dev_datasets.AUDIO_FILE_NAME
    dev_label = dev_datasets.KEY

    train_label_onehot = []
    for i in train_label:
        if (i == "bonafide"):
            train_label_onehot.append(1)
        else:
            train_label_onehot.append(0)

    eval_label_onehot = []
    for i in eval_label:
        if (i == "bonafide"):
            eval_label_onehot.append(1)
        else:
            eval_label_onehot.append(0)

    dev_label_onehot = []
    for i in dev_label:
        if (i == "bonafide"):
            dev_label_onehot.append(1)
        else:
            dev_label_onehot.append(0)


    #print(len(train_label))

    train_label_onehot = np.array(train_label_onehot)
    #print(train_label_onehot.shape)


    eval_label_onehot = np.array(eval_label_onehot)

    dev_label_onehot = np.array(dev_label_onehot)

    train_label_onehot = keras.utils.to_categorical(train_label_onehot)
    eval_label_onehot = keras.utils.to_categorical(eval_label_onehot)
    dev_label_onehot = keras.utils.to_categorical(dev_label_onehot)
    #print("---------------onehot--------------------")
    #print(train_label_onehot.shape)

    train_data = np.load("train.npy")
    dev_data = np.load("dev.npy")

    model = network.build_model((1025, 162, 1))
    model.load_weights("final.h5")
    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    # model.fit_generator(train_generate(train_file, train_label_onehot), steps_per_epoch=len(train_file)/64, epochs=150, validation_data=eval_generate(eval_file, eval_label_onehot), validation_steps=len(eval_file)/64, max_queue_size=1, workers=1)
    model.fit(train_data, train_label_onehot, batch_size=64, epochs=6, validation_data=(dev_data, dev_label_onehot), shuffle=True)

    model.save_weights("LeNet.h5")