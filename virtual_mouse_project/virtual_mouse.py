
import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

# Smoothing setup
prev_x, prev_y = 0, 0
smoothing = 5

click_delay = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = hand_landmarks.landmark

            # Get landmark positions
            index_tip = lm_list[8]
            thumb_tip = lm_list[4]
            middle_tip = lm_list[12]
            wrist = lm_list[0]

            # Cursor movement with index finger
            x = int(index_tip.x * screen_w)
            y = int(index_tip.y * screen_h)

            # Smoothing movement
            curr_x = prev_x + (x - prev_x) / smoothing
            curr_y = prev_y + (y - prev_y) / smoothing
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Click detection (thumb + index finger pinch)
            x1, y1 = int(thumb_tip.x * screen_w), int(thumb_tip.y * screen_h)
            x2, y2 = int(index_tip.x * screen_w), int(index_tip.y * screen_h)
            click_distance = math.hypot(x2 - x1, y2 - y1)

            if click_distance < 40 and click_delay == 0:
                pyautogui.click()
                click_delay = 10

            # Drag gesture (index and middle finger close)
            x3, y3 = int(middle_tip.x * screen_w), int(middle_tip.y * screen_h)
            drag_distance = math.hypot(x3 - x2, y3 - y2)
            if drag_distance < 40:
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()

            # Scroll gesture using hand tilt (wrist y and index y)
            wrist_y = wrist.y
            index_y = index_tip.y
            diff_y = index_y - wrist_y

            if diff_y < -0.05:
                pyautogui.scroll(20)
            elif diff_y > 0.05:
                pyautogui.scroll(-20)

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Countdown to reset click delay
    if click_delay > 0:
        click_delay -= 1

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
