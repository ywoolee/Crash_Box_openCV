# Crash_Box_openCV


### 🖥 About project
- Crash_Box_openCV is an interactive, game that utilizes computer vision to track hand movements and control paddle. The game starts with a ball bouncing around the screen, and the player's goal is to keep the ball in play by hitting it with a virtual paddle controlled by hand on camera. Collect x2 item to boost your score and avoid game-over by preventing the ball from hitting the left edge of the screen.



### ⚙️ Development environment
- `python 3.11.5`
- **IDE** : Spyder

### 📋 Before exqution
 1. prepare camera
if you want to use this you need to prepare camera with resolution of at least "1280 * 720"


 2. install tools
install using your IDE's terminal
    ```python
    pip install opencv-python
    pip install midiapipe
    pip install numpy
    pip install cvzone
    ```
    
4. Prepare the Resources:
Ensure the following images are present in the Resources directory:
    ```
    Background.png
    Ball.png
    board.png
    Box.png
    x2.png
    StartScreen.png
    GameOver.png
    ```
    
4. Enjoy Game

 
### 🎮play exemple
1. If you run, this start screen will appear. Press SPACE to start the game
    ![main](https://github.com/ywoolee/Crash_Box_openCV/assets/68912105/068fb70e-407e-4ecd-a148-bfa809421a49)
2. If you move the paddle on the left side of the game screen up and down using either your right or left hand and hit the ball to break the box, the score will increase
   2-1. The item is X2, but if you eat it, your score will be twice as high as your score in a certain period of time
   2-2. If the ball hits the left wall, the game ends
    **If you use more than two hands, the paddle moves based on the recognized hand
    ![hand_detect](https://github.com/ywoolee/Crash_Box_openCV/assets/68912105/40f0afaf-18ea-4643-8f57-be921cf04d5e)
3. If you press R on the end screen, you'll go back to the start screen
![end](https://github.com/ywoolee/Crash_Box_openCV/assets/68912105/eb6d0529-f4ad-46c7-81c9-7a294d07d0b8)

    **If you want to exit, press ESC to exit


#### reference
Hand Tracking Module
https://vrworld.tistory.com/12

Overlay At Real Time
https://www.toolify.ai/ko/ai-news-kr/opencv-png-1119795

