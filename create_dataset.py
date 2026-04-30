import os
import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)

DATA_DIR = './data'
data = []
labels = []

with vision.HandLandmarker.create_from_options(options) as landmarker:
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            results = landmarker.detect(mp_image)

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]

                for landmark in hand_landmarks:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                x_range = max(x_) - min(x_)
                y_range = max(y_) - min(y_)

                for landmark in hand_landmarks:
                    data_aux.append((landmark.x - min(x_)) / x_range if x_range > 0 else 0)
                    data_aux.append((landmark.y - min(y_)) / y_range if y_range > 0 else 0)

                data.append(data_aux)
                labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset created: {} samples across {} classes".format(len(data), len(set(labels))))