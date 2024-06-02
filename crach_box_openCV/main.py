import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import random
import time

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    imgBackground = cv2.imread("Resources/Background.png")
    imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
    imgBoard = cv2.imread("Resources/board.png", cv2.IMREAD_UNCHANGED)
    imgBox = cv2.imread("Resources/Box.png", cv2.IMREAD_UNCHANGED)
    imgItem = cv2.imread("Resources/x2.png", cv2.IMREAD_UNCHANGED)

    def generate_new_box_position():
        new_x = random.randint(400, 1185)
        new_y = random.randint(0, 410)
        return new_x, new_y

    def generate_new_item_position():
        new_x = random.randint(400, 1185)
        new_y = random.randint(0, 410)
        return new_x, new_y

    x_box, y_box = generate_new_box_position()
    x_item, y_item = generate_new_item_position()

    detector = HandDetector(detectionCon=0.8, maxHands=1)

    ballPos = [100, 100]
    base_speed = 15
    speed = base_speed
    direction = np.array([1, 1])
    direction = direction / np.linalg.norm(direction) * speed
    score = 0
    last_box_time = time.time()
    last_speed_increase_time = time.time()
    item_spawn_time = time.time()
    item_lifespan = 10
    item_effect_end_time = None
    box_visible = True
    item_visible = False
    score_multiplier = 1

    def maintain_speed(direction, speed):
        norm = np.linalg.norm(direction)
        if norm == 0:
            return direction
        return (direction / norm) * speed

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)

        hands, img = detector.findHands(img, flipType=False)
        img = cv2.addWeighted(img, 0.05, imgBackground, 0.95, 0)
        img = cvzone.overlayPNG(img, imgBall, tuple(map(int, ballPos)))

        current_time = time.time()
        if current_time - last_box_time > 0.5:
            box_visible = True
            img = cvzone.overlayPNG(img, imgBox, (x_box, y_box))
        else:
            box_visible = False

        if current_time - item_spawn_time > 30:
            item_visible = True
            x_item, y_item = generate_new_item_position()
            item_spawn_time = current_time

        if item_visible and current_time - item_spawn_time > item_lifespan:
            item_visible = False

        if item_visible:
            img = cvzone.overlayPNG(img, imgItem, (x_item, y_item))

        if current_time - last_speed_increase_time >= 1:
            speed += 0.05
            direction = maintain_speed(direction, speed)
            last_speed_increase_time = current_time

        if hands:
            for hand in hands:
                x, y, w, h = hand['bbox']
                h_board, w_board, _ = imgBoard.shape
                y_board = y - h_board // 2
                y_board = np.clip(y_board, 10, 405)

                board_rect = [20, y_board, w_board, h_board]

                ball_rect = [ballPos[0], ballPos[1], imgBall.shape[1], imgBall.shape[0]]

                if (ball_rect[1] < board_rect[1] + board_rect[3] and ball_rect[1] + ball_rect[3] > board_rect[1]):
                    if ball_rect[0] < board_rect[0] + board_rect[2] and ball_rect[0] + ball_rect[2] > board_rect[0]:
                        hit_position = (ball_rect[1] + ball_rect[3] / 2) - board_rect[1]
                        normalized_hit_position = (hit_position / board_rect[3]) - 0.5
                        direction[1] = direction[1] + normalized_hit_position * 2
                        direction[0] = -direction[0]
                        direction = maintain_speed(direction, speed)

                if (hand['type'] == "Left") or (hand['type'] == "Right"):
                    img = cvzone.overlayPNG(img, imgBoard, (20, y_board))

        if ballPos[1] >= 470 or ballPos[1] <= 0:
            direction[1] = -direction[1]
            direction = maintain_speed(direction, speed)
        if ballPos[0] >= 1242 or ballPos[0] <= 20:
            direction[0] = -direction[0]
            direction = maintain_speed(direction, speed)

        ball_rect = [ballPos[0], ballPos[1], imgBall.shape[1], imgBall.shape[0]]

        if box_visible:
            box_rect = [x_box, y_box, imgBox.shape[1], imgBox.shape[0]]
            if (ball_rect[1] < box_rect[1] + box_rect[3] and ball_rect[1] + ball_rect[3] > box_rect[1]):
                if ball_rect[0] < box_rect[0] + box_rect[2] and ball_rect[0] + ball_rect[2] > box_rect[0]:
                    if ball_rect[1] + ball_rect[3] - direction[1] < box_rect[1] or ball_rect[1] - direction[1] > box_rect[1] + box_rect[3]:
                        direction[1] = -direction[1]
                    if ball_rect[0] + ball_rect[2] - direction[0] < box_rect[0] or ball_rect[0] - direction[0] > box_rect[0] + box_rect[2]:
                        direction[0] = -direction[0]
                    score += 1 * score_multiplier
                    print(score)
                    last_box_time = time.time()
                    x_box, y_box = generate_new_box_position()
                    direction = maintain_speed(direction, speed)

        if item_visible:
            item_rect = [x_item, y_item, imgItem.shape[1], imgItem.shape[0]]
            if (ball_rect[1] < item_rect[1] + item_rect[3] and ball_rect[1] + ball_rect[3] > item_rect[1]):
                if ball_rect[0] < item_rect[0] + item_rect[2] and ball_rect[0] + ball_rect[2] > item_rect[0]:
                    item_visible = False
                    item_effect_end_time = current_time + 20
                    score_multiplier = 2
                    print("Item Collected! Score is doubled for 20 seconds.")

        if item_effect_end_time and current_time > item_effect_end_time:
            score_multiplier = 1
            item_effect_end_time = None
            print("Item effect ended. Score multiplier restored.")

        if item_effect_end_time and current_time <= item_effect_end_time:
            cv2.putText(img, "Double Score Active!", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        ballPos[0] += direction[0]
        ballPos[1] += direction[1]
        
        if 0 <= score <=9:
            cv2.putText(img, str(score) , (610, 690), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
        elif 10 <= score <= 99:  
            cv2.putText(img, str(score) , (580, 690), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
        elif score >= 100:
            cv2.putText(img, str(score) , (545, 690), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
        
        

        cv2.imshow("Image", img)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
