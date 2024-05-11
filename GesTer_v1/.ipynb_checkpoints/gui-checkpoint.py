import pickle
import cv2
import os
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class PresentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Presentation App")
        
        # Load model
        self.model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = self.model_dict['model']
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.h_small, self.w_small = int(120 * 1), int(213 * 1)  # height and width of small image
        
        # Variables
        self.imgNum = 0
        self.gestureActivated = False
        self.gestureCounter = 0
        self.gestureDelay = 30
        self.annotations = [[]]
        self.annotationNumber = -1
        self.annotationStart = False
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        
        # Labels dictionary
        self.labels_dict = {"One": 'Draw', "Two": 'Pointer', "Three": 'Erase', "Four": 4, "Five":5 , "ThumbsUp": 'Previous', "Fabulous": 'Next'}
        
        # GUI Elements
        self.select_folder_button = tk.Button(self.root, text="Select Presentation Folder", command=self.select_folder)
        self.select_folder_button.pack(pady=10)
        
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()
        
        self.quit_button = tk.Button(self.root, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)
        
        self.update_frame()
        
        # Resize the window to fit the screen
        self.root.attributes('-fullscreen', True)
    
    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folderPath = folder_selected
            self.pathImages = sorted(os.listdir(self.folderPath), key=len)
            print(self.pathImages)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if hasattr(self, 'pathImages'):
                pathFullImage = os.path.join(self.folderPath, self.pathImages[self.imgNum])
                imgCurr = cv2.imread(pathFullImage)
                imgCurr = cv2.resize(imgCurr, (frame.shape[1], frame.shape[0]))
                
                results = self.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks and not self.gestureActivated:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    data_aux = []
                    x_ = []
                    y_ = []
                    indexFinger = (0, 0)
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            if i == 8:
                                x_idx = hand_landmarks.landmark[i].x
                                y_idx = hand_landmarks.landmark[i].y
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
                        x1 = int(min(x_) * frame.shape[1]) - 10
                        y1 = int(min(y_) * frame.shape[0]) - 10
                        
                        x2 = int(max(x_) * frame.shape[1]) - 10
                        y2 = int(max(y_) * frame.shape[0]) - 10
                        
                        x8 = int(x_idx * frame.shape[1]) - 10
                        y8 = int(y_idx * frame.shape[0]) - 10
                        indexFinger = (x8, y8)
                        
                        prediction = self.model.predict([np.asarray(data_aux)])
                        predicted_character = self.labels_dict[prediction[0]]
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, str(predicted_character), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 200), 3, cv2.LINE_AA)
                        
                        if predicted_character == 'Next' and self.imgNum < len(self.pathImages) - 1:
                            self.annotationStart = False
                            self.gestureActivated = True
                            self.imgNum += 1
                        if predicted_character == 'Previous' and self.imgNum > 0:
                            self.annotationStart = False
                            self.gestureActivated = True
                            self.imgNum -= 1
                        if predicted_character == 'Pointer':
                            self.annotationStart = False
                            cv2.circle(imgCurr, indexFinger, 5, (0, 0, 255), cv2.FILLED)
                        if predicted_character == 'Draw':
                            if not self.annotationStart:
                                self.annotationStart = True
                                self.annotationNumber += 1
                                self.annotations.append([])
                            cv2.circle(imgCurr, indexFinger, 5, (0, 0, 255), cv2.FILLED)
                            self.annotations[self.annotationNumber].append(indexFinger)
                        if predicted_character == 'Erase':
                            self.annotationStart = False
                            self.annotations = [[]]
                            self.annotationNumber = -1
                            self.annotationStart = False
                
                if self.gestureActivated:
                    self.gestureCounter += 1
                    if self.gestureCounter > self.gestureDelay:
                        self.gestureCounter = 0
                        self.gestureActivated = False
                
                for i in range(len(self.annotations)):
                    for j in range(len(self.annotations[i])):
                        if j != 0:
                            cv2.line(imgCurr, self.annotations[i][j-1], self.annotations[i][j], (0, 0, 200), 5)
                
                imgSmall = cv2.resize(frame, (self.w_small, self.h_small))
                h, w, _ = imgCurr.shape
                imgCurr[0:self.h_small, w-self.w_small:w] = imgSmall
                
                imgCurr = cv2.cvtColor(imgCurr, cv2.COLOR_BGR2RGB)
                imgTk = cv2.resize(imgCurr, (640, 480))
                
                imgTk = Image.fromarray(imgTk)
                imgTk = ImageTk.PhotoImage(image=imgTk)
                
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgTk)
                self.canvas.imgTk = imgTk
            
        self.root.after(10, self.update_frame)

    def quit_app(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PresentationApp(root)
    root.mainloop()
