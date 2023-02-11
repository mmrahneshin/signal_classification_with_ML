import numpy as np
import tensorflow as tf

from cnn.select_features.main import select_feature
from cnn.features.initial import initialize_features
from cnn.Kfold import K_fold
from tensorflow.keras import  layers

def cnn(data, label):
    BATCH_SIZE = 256
    EPOCHS = 5
    input_size = data.shape

    train_index, test_index = K_fold(label, 5)
    x_features1 = initialize_features(data[0])
    x_features2 = initialize_features(data[1])
    x_features1 = select_feature(x_features1, label, train_index, test_index , 5)
    x_features2 = select_feature(x_features2, label, train_index, test_index , 5)
    x_features = np.array([x_features1, x_features2])

    print(x_features.shape)

    train_index = np.array(train_index[0])
    test_index = np.array(test_index[0])

    x_features = x_features[:,train_index,:]
    x_train = data[:,train_index,:]
    y_train = label[train_index]
    x_test = data[:,test_index,:]
    y_test = label[test_index]

    print(x_train.shape, x_test.shape)
    print(np.sum(y_test), np.sum(y_train))

    # train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_data = train_data.shuffle(100).batch(BATCH_SIZE)

    # test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # test_data = test_data.shuffle(100).batch(BATCH_SIZE)


    # inputs = tf.keras.Input(shape=input_size)
    # model = layers.Conv1D(filters = 20,kernel_size=(3,3))(inputs)
    # model = layers.MaxPool1D(pool_size = (2,2))(model)
    # model = layers.Flatten()(model)
    # model = layers.Dense(256, activation='relu')(model)
    # model = layers.Dense(64, activation='relu', name='feature')(model)
    # outputs = layers.Dense(2, name='classification', activation='softmax')(model)

    # model = tf.keras.Model(inputs, outputs)

    # base_learning_rate = 0.0001
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    #           loss={'classification': tf.keras.losses.CategoricalCrossentropy()},
    #           metrics=['accuracy'])
    # model.summary()


    # history = model.fit(train_data,
    #                 epochs=EPOCHS,
    #                 validation_data=test_data)