import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np


cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4, 720)
#640 / 480이 이 노트북 카메라의 최대크기


imgBall = cv2.imread("Resources/Ball.png",cv2.IMREAD_UNCHANGED)
imgBoard1 = cv2.imread("Resources/board1.png",cv2.IMREAD_UNCHANGED)
imgBoard2 = cv2.imread("Resources/board2.png",cv2.IMREAD_UNCHANGED)


# Hand Detector
detector = HandDetector(detectionCon=0.8,maxHands=2)

while True:
    
    _, img = cap.read()
    img = cv2.flip(img,1)
    
    # Find hands in the current frame
    hands, img = detector.findHands(img,flipType=False) #img지우면 그림 안뜸
    
    
    #overlaying the background image
    #cv2.addWeighted(img,0.2,imgBackground,0.8,0) ->0.2랑 0.8로 투명도 조절
    
    #check for hands
    if hands:
        for hand in hands:
            x,y,w,h = hand['bbox']
            h1, w1, _ = imgBoard1.shape
            y1 = y - h1//2
            y1 = np.clip(y1,20,415)
            if hand['type'] == "Left":
                cvzone.overlayPNG(img, imgBoard1,(20,y1))
    
    
    #Draw the ball
    cvzone.overlayPNG(img, imgBall,(100,100))
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()    

        
    