import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pickle
import numpy as np

# โหลด model
model = pickle.load(open("model.pkl","rb"))

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    sucess, image = cap.read()
    if not sucess:
        break

    image = cv2.flip(image,1)
    # change BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect_for_video(mp_image, int(time.time()*1000))

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        data = []
        for lm in hand:
            data.append(lm.x)
            data.append(lm.y)
        
        probs = model.predict_proba([data])[0]
        max_prob = max(probs)
        pred = model.classes_[probs.argmax()]

        if max_prob > 0.7:
            text = pred
            cv2.putText(image, text, (50,80),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        # else:
        #     text = "Not Found"
        # cv2.putText(image, text, (50,80),
        #                 cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)


        for lm in hand:
            h,w,_ = image.shape
            x,y = int(lm.x*w), int(lm.y*h)
            cv2.circle(image,(x,y),4,(255,255,0),-1)

    cv2.imshow("Alphabet Detect", image)
    
    # esc to exit 
    if cv2.waitKey(1)==27: 
        break

cap.release()
cv2.destroyAllWindows()
