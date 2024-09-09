import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import os

capture = cv2.VideoCapture(0)  
detector = HandDetector(maxHands=1)  

dir = "dataset"            #aafu la bhako folder ko name lakna tara aautai folder ma bha bana name lakna natra puri path dina 
letter='ka'                # change garna ka bata kha ma aafai la aani code lai phari run garna 
count=0 
location=os.path.join(dir,letter)
if not os.path.exists(location):
    os.makedirs(location)  

white = np.ones((400, 400), np.uint8) * 255         #sato background banu na lai

finger_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),                  # nadibata budi aaula sama
    (5, 6), (6, 7), (7, 8),                          # chori aaula ko
    (9, 10), (10, 11), (11, 12),                     # bicha ko aaula 
    (13, 14), (14, 15), (15, 16),                    # saili aaula ko
    (17, 18), (18, 19), (19, 20),                    # kanchi aaula
    (0, 5), (5, 9), (9, 13), (13, 17),               # hatkala haru
    (0, 17) ]                                        # nadibata kanchi aaula sama

while True:
    success, frame = capture.read()  
    frame = cv2.flip(frame, 1)                         # web camera bata lida mirror jasto kam garcha so flip gara ko
    hands, _ = detector.findHands(frame, draw=False)  
 
    if hands:
        hand = hands[0]                                #first hand lai lock gara ko aasla garda second hand ignore huncha 
        landmarks = hand['lmList']                     # yas la hat ko 21 ota point dincha jun Mediapipe ma define cha 
       
        # hat lai center mA Launu ko lagi 
        
        x_min = min([lm[0] for lm in landmarks])
        y_min = min([lm[1] for lm in landmarks])
        x_max = max([lm[0] for lm in landmarks])
        y_max = max([lm[1] for lm in landmarks])
        hand_width = x_max - x_min
        hand_height = y_max - y_min
        os_x = (400 - hand_width) // 2 - x_min
        os_y = (400 - hand_height) // 2 - y_min
 
        white.fill(255)#reset gara ko 

        # yasla line banucha joint garcha point haru
        
        for start, end in finger_connections:
            cv2.line(white, (landmarks[start][0] + os_x, landmarks[start][1] + os_y),
                     (landmarks[end][0] + os_x, landmarks[end][1] + os_y), (0, 0, 0), 3)

        # point ma circles banucha
        
        for i in range(21):
            cv2.circle(white, (landmarks[i][0] + os_x, landmarks[i][1] + os_y), 4, (0, 0, 225), -2)
    
    cv2.imshow("Hand Skeleton", white)
    cv2.imshow("Video Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        cv2.imwrite(f"{location}/{count}.jpg", white)                 # Save skeleton image
        count += 1
        print(f"Image saved: {count}")
    if key == 27:                                                     # ASCII VALUE OF ESC key is 27 
        break

capture.release()
cv2.destroyAllWindows()
