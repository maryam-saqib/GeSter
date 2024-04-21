# Testing the Model
import pickle

import cv2
import os
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

# Get the list of presentation images
folderPath='Presentation'
pathImages=sorted(os.listdir(folderPath),key=len)
print(pathImages)

# Variables
imgNum=0 # Image Number variable(will be used to move through slides)
h_small,w_small=int(120*1),int(213*1) # height and width of small image
gestureActivated=False
gestureCounter=0
gestureDelay=25

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {"One": 1,  "Two": 2, "Three": 3, "Four": 4, "Five": 5, "ThumbsUp": 6,"Fabulous": 7}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Import images
    pathFullImage=os.path.join(folderPath,pathImages[imgNum])
    imgCurr=cv2.imread(pathFullImage)
    imgCurr = cv2.resize(imgCurr, (frame.shape[1], frame.shape[0]))  # Resize slides image to match webcam frame size

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks and gestureActivated==False:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
        if len(data_aux) <= 42:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[prediction[0]]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, str(predicted_character), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            
            if predicted_character==5 and imgNum<len(pathImages)-1:
                gestureActivated=True
                imgNum+=1
            elif predicted_character==6 and imgNum>0:
                gestureActivated=True
                imgNum-=1

            print(predicted_character)


    # gesture activated iterations
    if gestureActivated:
        gestureCounter+=1
        if gestureCounter>gestureDelay:
            gestureCounter=0
            gestureActivated=False

    # Adding webcam on slides window
    imgSmall=cv2.resize(frame,(w_small,h_small))
    h,w,_=imgCurr.shape
    imgCurr[0:h_small,w-w_small:w]=imgSmall

    cv2.imshow('frame', frame)
    cv2.imshow('Slides',imgCurr) # display images
    
    # Exit if the user presses the close button (X) or 'q'
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Slides', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
