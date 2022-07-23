import numpy as np
import cv2
import librosa
import subprocess
from tensorflow.keras.models import *


def load_dataset(path):
    audios = []
    audio, sample_rate = librosa.load(path, res_type='kaiser_fast')
    # Extract mfcc
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    # Extract melspectrogram
    mel_features = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
    # In order to find out scaled feature, do mean of transpose of value
    mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
    mel_scaled_features = np.mean(mel_features.T, axis=0)
    audios.append(mfcc_scaled_features)
    audios.append(mel_scaled_features)
    return audios


def video_to_frames(path):
    # Start capturing the feed
    sec = 1
    vid_cap = cv2.VideoCapture(path)
    vid_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    has_frames, image = vid_cap.read()
    if has_frames:
        resulting_image = crop_faces(image)
    return resulting_image


def crop_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
    img = cv2.resize(face, (64, 64))
    resulting_image = np.array(img)/255.0
    # cv2.imshow("resulting_image", resulting_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return resulting_image


if __name__ == '__main__':
    command = "ffmpeg -i test/anger.avi -ab 160k -ac 2 -ar 44100 -vn test/audio.wav"
    subprocess.call(command, shell=True)

    audio_path = "test/audio.wav"
    video_path = "test/anger.avi"
    audio_data = load_dataset(audio_path)

    audio_data = np.array(audio_data)
    audio_data = audio_data.reshape(1, 80, 1)

    image_data = video_to_frames(video_path)
    image_data = np.array(image_data)
    image_data = image_data.reshape(1, 64, 64, 3, 1)

    # Load model
    savedModel = load_model('EmotionRecognitionModel.h5')
    # savedModel.summary()
    print('Model Loaded!')

    prediction = savedModel.predict([image_data, audio_data])
    prediction = prediction.argmax(axis=1)
    prediction = prediction.astype(int).flatten()
    if prediction[0] == 0:
        print("Sadness")
    elif prediction[0] == 1:
        print("Surprise")
    elif prediction[0] == 2:
        print("Neutral")
    elif prediction[0] == 3:
        print("Happiness")
    elif prediction[0] == 4:
        print("Anger")
    elif prediction[0] == 5:
        print("Disgust")
    elif prediction[0] == 6:
        print("Fear")

