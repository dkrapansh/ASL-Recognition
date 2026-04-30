import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)

labels_dict = {i: letter for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

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

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            prediction = model.predict([np.asarray(data_aux)])
            prediction_proba = model.predict_proba([np.asarray(data_aux)])
            confidence = np.max(prediction_proba)

            if confidence > 0.6:
                predicted_character = prediction[0]
            else:
                predicted_character = "?"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            label = '{} ({:.0f}%)'.format(predicted_character, confidence * 100)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()