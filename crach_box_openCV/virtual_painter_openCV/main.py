import cv2
import cvzone
import numpy as np
import random
import math
import mediapipe as mp


class HandDetector:
    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)

        return allHands, img

    def fingersUp(self, myHand):
        fingers = []
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, info, img


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    imgBackground = cv2.imread("Resources/Background.png")
    imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
    imgBoard = cv2.imread("Resources/board.png", cv2.IMREAD_UNCHANGED)
    imgBox = cv2.imread("Resources/Box.png", cv2.IMREAD_UNCHANGED)

    x_box = random.randint(400, 1185)
    y_box = random.randint(0, 410)

    detector = HandDetector(detectionCon=0.8, maxHands=1)

    ballPos = [100, 100]
    speedX = 15
    speedY = 15
    score = 0

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)

        hands, img = detector.findHands(img, flipType=False)
        h1, w1, _ = imgBoard.shape

        img = cv2.addWeighted(img, 0.05, imgBackground, 0.95, 0)

        if hands:
            for hand in hands:
                x, y, w, h = hand['bbox']
                h1, w1, _ = imgBoard.shape
                y1 = y - h1 // 2
                y1 = np.clip(y1, 10, 405)
                if (hand['type'] == "Left") or (hand['type'] == "Right"):
                    cvzone.overlayPNG(img, imgBoard, (20, y1))
                    if 20 < ballPos[0] < 25 + w1 and y1 < ballPos[1] < y1 + h1:
                        speedX = -speedX
                        ballPos[0] += 5

        if ballPos[1] >= 470 or ballPos[1] <= 0:
            speedY = -speedY
        if ballPos[0] >= 1242 or ballPos[0] <= 20:
            speedX = -speedX

        ball_rect = [ballPos[0], ballPos[1], imgBall.shape[1], imgBall.shape[0]]
        box_rect = [x_box, y_box, imgBox.shape[1], imgBox.shape[0]]

        if (ball_rect[1] < box_rect[1] + box_rect[3] and
            ball_rect[1] + ball_rect[3] > box_rect[1]):
            if ball_rect[0] < box_rect[0] + box_rect[2] and ball_rect[0] + ball_rect[2] > box_rect[0]:
                if ball_rect[1] + ball_rect[3] - speedY < box_rect[1] or ball_rect[1] - speedY > box_rect[1] + box_rect[3]:
                    speedY = -speedY
                if ball_rect[0] + ball_rect[2] - speedX < box_rect[0] or ball_rect[0] - speedX > box_rect[0] + box_rect[2]:
                    speedX = -speedX
                score += 1
                print(score)

        ballPos[0] += speedX
        ballPos[1] += speedY

        cvzone.overlayPNG(img, imgBall, ballPos)
        cvzone.overlayPNG(img, imgBox, (x_box, y_box))

        cv2.imshow("Image", img)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

    