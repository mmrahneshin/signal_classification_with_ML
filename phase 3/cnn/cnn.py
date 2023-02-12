import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from cnn.select_features.main import select_feature
from cnn.features.initial import initialize_features
from cnn.Kfold import K_fold
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix





def cnn1(data, label):

    train_index, test_index = K_fold(label, 5)
    x_features1 = initialize_features(data[0])
    x_features2 = initialize_features(data[1])
    x_features1 = select_feature(x_features1, label, train_index, test_index , 5)
    x_features2 = select_feature(x_features2, label, train_index, test_index , 5)
    x_features = np.array([x_features1, x_features2])

    print(x_features.shape)

    train_index = np.array(train_index[0])
    test_index = np.array(test_index[0])

    data = data.reshape(data.shape[1], data.shape[2], data.shape[0])
    x_features = x_features.reshape(x_features.shape[1], x_features.shape[2], x_features.shape[0])
    
    x_features_train = x_features[train_index,:,:]
    x_features_test = x_features[test_index,:,:]

    x_train = data[train_index,:,:]
    y_train = label[train_index]
    x_test = data[test_index,:,:]
    y_test = label[test_index]
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    print(x_train.shape, x_features_train.shape, y_train_cat.shape)
    print(np.sum(y_test), np.sum(y_train))

    EPOCHS = 5
    input_size = [data.shape[1], data.shape[2]]


    inputs = tf.keras.Input(shape=input_size)
    features_inputs = tf.keras.Input(shape=(x_features_train.shape[1], x_features_train.shape[2]), name="features_inputs")
    flatten_features = layers.Flatten()(features_inputs)
    model = layers.Conv1D(filters = 256,kernel_size=5)(inputs)
    model = layers.MaxPooling1D(pool_size=2)(model)
    model = layers.Conv1D(filters = 128,kernel_size=5)(model)
    model = layers.MaxPooling1D(pool_size=2)(model)
    model = layers.Conv1D(filters = 64,kernel_size=5)(model)
    model = layers.MaxPooling1D(pool_size=2)(model)
    model = layers.Flatten()(model)
    model = layers.Dense(32, activation='relu')(model)

    concatenated = layers.Concatenate()([model, flatten_features])
    model = layers.Dense(16, activation='relu', name='feature')(concatenated)
    outputs = layers.Dense(2, name='classification', activation='softmax')(model)

    model = tf.keras.Model([inputs, features_inputs], outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss={'classification': tf.keras.losses.CategoricalCrossentropy()},
              metrics=['accuracy'])
    model.summary()

    model.fit([x_train, x_features_train],y_train_cat,
               epochs=EPOCHS,shuffle=True
               ,verbose=2
               )

    model_loss, model_accuracy = model.evaluate([x_test, x_features_test], y_test_cat, verbose=2)
    print(f" Loss: {model_loss},Accuracy: {model_accuracy}")
     
    y_pred = model.predict([x_test, x_features_test])
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test_cat, axis=1)
    acc_score = accuracy_score(y_test,y_pred)
    rec_score = recall_score(y_test,y_pred)
    prec_score = precision_score(y_test,y_pred)


    print("accuracy score : ", acc_score)
    print("recall score : ", rec_score)
    print("precision score : ", prec_score)

    cm = confusion_matrix(y_test, y_pred)

    false_alarm_rate = cm[0][1] / (cm[0][1] + cm[0][0])
    print("false alarm rate: ", false_alarm_rate)










def cnn2(data, label):

    train_index, test_index = K_fold(label, 5)
    x_features1 = initialize_features(data[0])
    x_features2 = initialize_features(data[1])
    x_features1 = select_feature(x_features1, label, train_index, test_index , 5)
    x_features2 = select_feature(x_features2, label, train_index, test_index , 5)
    x_features = np.array([x_features1, x_features2])

    print(x_features.shape)

    train_index = np.array(train_index[0])
    test_index = np.array(test_index[0])

    data = data.reshape(data.shape[1], data.shape[2], data.shape[0])
    x_features = x_features.reshape(x_features.shape[1], x_features.shape[2], x_features.shape[0])
    
    x_features_train = x_features[train_index,:,:]
    x_features_test = x_features[test_index,:,:]

    x_train = data[train_index,:,:]
    y_train = label[train_index]
    x_test = data[test_index,:,:]
    y_test = label[test_index]
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    print(x_train.shape, x_features_train.shape, y_train_cat.shape)
    print(np.sum(y_test), np.sum(y_train))

    EPOCHS = 5
    input_size = [data.shape[1], data.shape[2]]


    inputs = tf.keras.Input(shape=input_size)
    features_inputs = tf.keras.Input(shape=(x_features_train.shape[1], x_features_train.shape[2]), name="features_inputs")
    flatten_features = layers.Flatten()(features_inputs)
    model = layers.Conv1D(filters = 256,kernel_size=5)(inputs)
    model = layers.MaxPooling1D(pool_size=2)(model)
    model = layers.Conv1D(filters = 128,kernel_size=5)(model)
    model = layers.MaxPooling1D(pool_size=2)(model)
    model = layers.Conv1D(filters = 64,kernel_size=5)(model)
    model = layers.MaxPooling1D(pool_size=2)(model)
    model = layers.Flatten()(model)
    model = layers.Dense(32, activation='relu')(model)
    model = layers.Dense(16, activation='relu')(model)

    concatenated = layers.Concatenate()([model, flatten_features])
    model = layers.Dense(8, activation='relu', name='feature')(concatenated)
    outputs = layers.Dense(2, name='classification', activation='softmax')(model)

    model = tf.keras.Model([inputs, features_inputs], outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss={'classification': tf.keras.losses.CategoricalCrossentropy()},
              metrics=['accuracy'])
    model.summary()

    model.fit([x_train, x_features_train],y_train_cat,
               epochs=EPOCHS,shuffle=True
               ,verbose=2
               )

    model_loss, model_accuracy = model.evaluate([x_test, x_features_test], y_test_cat, verbose=2)
    print(f" Loss: {model_loss},Accuracy: {model_accuracy}")

    
    y_pred = model.predict([x_test, x_features_test])
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test_cat, axis=1)
    acc_score = accuracy_score(y_test,y_pred)
    rec_score = recall_score(y_test,y_pred)
    prec_score = precision_score(y_test,y_pred)

    print("accuracy score : ", acc_score)
    print("recall score : ", rec_score)
    print("precision score : ", prec_score)

    
    cm = confusion_matrix(y_test, y_pred)

    false_alarm_rate = cm[0][1] / (cm[0][1] + cm[0][0])
    print("false alarm rate: ", false_alarm_rate)


















    
def cnn3(data, label):

    train_index, test_index = K_fold(label, 5)
    x_features1 = initialize_features(data[0])
    x_features2 = initialize_features(data[1])
    x_features1 = select_feature(x_features1, label, train_index, test_index , 5)
    x_features2 = select_feature(x_features2, label, train_index, test_index , 5)
    x_features = np.array([x_features1, x_features2])

    print(x_features.shape)

    train_index = np.array(train_index[0])
    test_index = np.array(test_index[0])

    data = data.reshape(data.shape[1], data.shape[2], data.shape[0])
    x_features = x_features.reshape(x_features.shape[1], x_features.shape[2], x_features.shape[0])
    
    x_features_train = x_features[train_index,:,:]
    x_features_test = x_features[test_index,:,:]

    x_train = data[train_index,:,:]
    y_train = label[train_index]
    x_test = data[test_index,:,:]
    y_test = label[test_index]
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    print(x_train.shape, x_features_train.shape, y_train_cat.shape)
    print(np.sum(y_test), np.sum(y_train))

    EPOCHS = 5
    input_size = [data.shape[1], data.shape[2]]


    inputs = tf.keras.Input(shape=input_size)
    features_inputs = tf.keras.Input(shape=(x_features_train.shape[1], x_features_train.shape[2]), name="features_inputs")
    flatten_features = layers.Flatten()(features_inputs)
    model = layers.Conv1D(filters = 256,kernel_size=5)(inputs)
    model = layers.MaxPooling1D(pool_size=2)(model)
    model = layers.Conv1D(filters = 128,kernel_size=5)(model)
    model = layers.MaxPooling1D(pool_size=2)(model)
    model = layers.Conv1D(filters = 64,kernel_size=5)(model)
    model = layers.MaxPooling1D(pool_size=2)(model)
    model = layers.Conv1D(filters = 32,kernel_size=5)(model)
    model = layers.Flatten()(model)
    model = layers.Dense(32, activation='relu')(model)

    concatenated = layers.Concatenate()([model, flatten_features])
    model = layers.Dense(16, activation='relu', name='feature')(concatenated)
    outputs = layers.Dense(2, name='classification', activation='softmax')(model)

    model = tf.keras.Model([inputs, features_inputs], outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss={'classification': tf.keras.losses.CategoricalCrossentropy()},
              metrics=['accuracy'])
    model.summary()

    model.fit([x_train, x_features_train],y_train_cat,
               epochs=EPOCHS,shuffle=True
               ,verbose=2
               )

    model_loss, model_accuracy = model.evaluate([x_test, x_features_test], y_test_cat, verbose=2)
    print(f" Loss: {model_loss},Accuracy: {model_accuracy}")

    
    y_pred = model.predict([x_test, x_features_test])
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test_cat, axis=1)
    acc_score = accuracy_score(y_test,y_pred)
    rec_score = recall_score(y_test,y_pred)
    prec_score = precision_score(y_test,y_pred)

    print("accuracy score : ", acc_score)
    print("recall score : ", rec_score)
    print("precision score : ", prec_score)

    
    cm = confusion_matrix(y_test, y_pred)

    false_alarm_rate = cm[0][1] / (cm[0][1] + cm[0][0])
    print("false alarm rate: ", false_alarm_rate)
