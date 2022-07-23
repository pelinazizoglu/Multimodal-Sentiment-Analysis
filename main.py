import numpy as np
import pandas as pd
import glob
import cv2
import os
import time
import librosa

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


def load_dataset():
    audios = []
    labels = []
    audio_filenames = []
    path = 'AudioData/'
    people = ["DC", "JE", "JK", "KL"]
    for person in people:
        new_path = path + person
        for audio_filename in sorted(glob.glob(os.path.join(new_path, '*.wav'))):
            # Load the file (audio)
            audio, sample_rate = librosa.load(audio_filename, res_type='kaiser_fast')
            # Extract mfcc
            mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            # Extract melspectrogram
            mel_features = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
            # In order to find out scaled feature, do mean of transpose of value
            mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
            mel_scaled_features = np.mean(mel_features.T, axis=0)
            audios.append(mfcc_scaled_features)
            audios.append(mel_scaled_features)
            audio_filenames.append(audio_filename)
            # Add labels
            if audio_filename.split("/")[2][0:2] == "sa":  # sadness = 0
                labels.append(0)
            elif audio_filename.split("/")[2][0:2] == "su":  # surprise = 1
                labels.append(1)
            elif audio_filename.split("/")[2][0] == "n":  # neutral = 2
                labels.append(2)
            elif audio_filename.split("/")[2][0] == "h":  # happiness = 3
                labels.append(3)
            elif audio_filename.split("/")[2][0] == "a":  # anger = 4
                labels.append(4)
            elif audio_filename.split("/")[2][0] == "d":  # disgust = 5
                labels.append(5)
            elif audio_filename.split("/")[2][0] == "f":  # fear = 6
                labels.append(6)
    return audios, audio_filenames, labels


def video_to_frames():
    labels = []
    image_data = []
    image_label = []
    path = 'VideoData/'
    people = ["DC", "JE", "JK", "KL"]
    for person in people:
        new_path = path + person
        for video_filename in sorted(glob.glob(os.path.join(new_path, '*.avi'))):
            if video_filename.split("/")[2][0:2] == "sa":  # sadness = 0
                labels.append(0)
            elif video_filename.split("/")[2][0:2] == "su":  # surprise = 1
                labels.append(1)
            elif video_filename.split("/")[2][0] == "n":  # neutral = 2
                labels.append(2)
            elif video_filename.split("/")[2][0] == "h":  # happiness = 3
                labels.append(3)
            elif video_filename.split("/")[2][0] == "a":  # anger = 4
                labels.append(4)
            elif video_filename.split("/")[2][0] == "d":  # disgust = 5
                labels.append(5)
            elif video_filename.split("/")[2][0] == "f":  # fear = 6
                labels.append(6)
            # Start capturing the feed
            sec = 1
            vid_cap = cv2.VideoCapture(video_filename)
            vid_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            has_frames, image = vid_cap.read()
            if has_frames:
                resulting_image = crop_faces(image)
                image_data.append(resulting_image)
                image_label.append(labels[-1])
    return image_data, image_label


def crop_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
    # Crop faces
    img = cv2.resize(face, (64, 64))
    resulting_image = np.array(img)/255.0
    # cv2.imshow("resulting_image", resulting_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return resulting_image


if __name__ == '__main__':
    audio_data, filenames, labels = load_dataset()
    print(len(audio_data))

    audio_data = np.array(audio_data)
    audio_data = audio_data.reshape(480, 80, 1)

    image_data, image_label = video_to_frames()
    image_data = np.array(image_data)
    image_data = image_data.reshape(480, 64, 64, 3, 1)

    lb = LabelEncoder()
    labels = to_categorical(lb.fit_transform(image_label))

    images_shuffled, audios_shuffled, labels_shuffled = shuffle(image_data, audio_data, labels, random_state=0)

    video_features_train, video_features_val, audio_features_train, audio_features_val, labels_train, labels_val = \
        train_test_split(images_shuffled, audios_shuffled, labels_shuffled, test_size=0.3, random_state=42,
                         shuffle=True)
    video_features_val, video_features_test, audio_features_val, audio_features_test, labels_val, labels_test = \
        train_test_split(video_features_val, audio_features_val, labels_val, test_size=0.5, random_state=42,
                         shuffle=True)

    # Video model
    input_video = Input(shape=(64, 64, 3, 1))

    conv = ConvLSTM2D(64, 3, activation='relu', padding='same')(input_video)
    drop = Dropout(0.4)(conv)

    conv = Convolution2D(32, 3, activation='relu', padding='same')(drop)

    conv = Convolution2D(16, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01),
                         bias_regularizer=l2(0.01))(conv)
    pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv)
    drop = Dropout(0.4)(pool)

    output_video = Flatten()(drop)

    # Audio Model
    input_audio = Input(shape=(80, 1))

    conv = Convolution1D(96, 10, activation='relu')(input_audio)
    drop = Dropout(0.4)(conv)
    conv = Convolution1D(128, 10, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(drop)
    pool = MaxPooling1D(2)(conv)
    drop = Dropout(0.4)(pool)

    output_audio = Flatten()(drop)

    # Concatenation
    concatenate = keras.layers.Concatenate()([output_video, output_audio])
    final_layer = Dense(128, activation='relu')(concatenate)
    y_predict_emotions = Dense(7, activation='softmax')(final_layer)
    model = Model(inputs=[input_video, input_audio], outputs=[y_predict_emotions])
    print(model.summary())
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    history = model.fit([video_features_train, audio_features_train], labels_train,
                        epochs=60,
                        batch_size=32,
                        validation_data=([video_features_val, audio_features_val], labels_val))
    end = time.time()
    train_time_avg = (end - start) / 120

    start = time.time()
    score = model.evaluate([video_features_test, audio_features_test], labels_test)
    end = time.time()
    test_time = end - start
    print("Test loss of the model is - ", score[0])
    print("Test accuracy of the model is - ", score[1] * 100, "%")

    predictions = model.predict([video_features_test, audio_features_test])
    predictions = predictions.argmax(axis=1)
    predictions = predictions.astype(int).flatten()
    predictions = (lb.inverse_transform(predictions))
    actual = labels_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (lb.inverse_transform(actual))
    cm = confusion_matrix(actual, predictions, normalize='true')
    conf_matrix = pd.DataFrame(cm, index=None, columns=None)
    conf_matrix.to_csv(r'confusion_matrix.csv', index=None, header=False)

    print(classification_report(actual, predictions, target_names=['sa', 'su', 'n', 'h', 'a', 'd', 'f']))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.ylim(0, 10)
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    # Save model
    model.save('EmotionRecognitionModel5.h5')
    print('Model Saved!')
