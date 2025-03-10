import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from collections import deque

# Initialize Mediapipe modules
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Setup camera
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# Initialize variables for drawing mode
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_index = 0

paint_window = np.zeros((471, 636, 3)) + 255
paint_window = cv2.rectangle(paint_window, (40, 1), (140, 65), (0, 0, 0), 2)
paint_window = cv2.rectangle(paint_window, (160, 1), (255, 65), (255, 0, 0), 2)
paint_window = cv2.rectangle(paint_window, (275, 1), (370, 65), (0, 255, 0), 2)
paint_window = cv2.rectangle(paint_window, (390, 1), (485, 65), (0, 0, 255), 2)
paint_window = cv2.rectangle(paint_window, (505, 1), (600, 65), (0, 255, 255), 2)

cv2.putText(paint_window, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_window, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_window, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_window, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_window, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

mode = "draw"  # Initial mode
running = True  # Flag to control the running of the program

while running:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame.shape

    # Process hands
    hand_results = hands.process(frame_rgb)

    left_hand_open = False
    right_index_open = False

    # Check hand landmarks for toggle mode and drawing
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand_landmarks.landmark]

            # Determine hand (left or right)
            label = hand_info.classification[0].label  # "Left" or "Right"

            if label == "Left":
                # Check if the left hand is open or closed
                left_fingers_up = sum(landmarks[i][1] < landmarks[i - 2][1] for i in [8, 12, 16, 20])
                left_hand_open = left_fingers_up > 3  # Left hand is open if more than 3 fingers are up

            elif label == "Right":
                # Check if the right index finger is open
                right_index_open = landmarks[8][1] < landmarks[6][1]  # Right index is open if y-coordinate is less

        # Toggle mode based on left hand (open left hand = "eye mode", closed left hand = "draw mode")
        if left_hand_open:
            mode = "eye"
        else:
            mode = "draw"

    if mode == "eye":
        # Eye-controlled mouse mode
        face_results = face_mesh.process(frame_rgb)

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                if id == 1:
                    screen_x = int(landmark.x * screen_w)
                    screen_y = int(landmark.y * screen_h)
                    pyautogui.moveTo(screen_x, screen_y)

            # Blink detection for click
            left_eye = [landmarks[145], landmarks[159]]
            if (left_eye[0].y - left_eye[1].y) < 0.004:
                pyautogui.click()

    elif mode == "draw" and right_index_open:
        # Drawing mode (only if right index finger is open)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand_landmarks.landmark]
                fore_finger = landmarks[8]  # Right index finger position

                if fore_finger[1] <= 65:  # Button area
                    if 40 <= fore_finger[0] <= 140:
                        bpoints = [deque(maxlen=1024)]
                        gpoints = [deque(maxlen=1024)]
                        rpoints = [deque(maxlen=1024)]
                        ypoints = [deque(maxlen=1024)]
                        blue_index = green_index = red_index = yellow_index = 0
                        paint_window[67:, :, :] = 255
                    elif 160 <= fore_finger[0] <= 255:
                        color_index = 0
                    elif 275 <= fore_finger[0] <= 370:
                        color_index = 1
                    elif 390 <= fore_finger[0] <= 485:
                        color_index = 2
                    elif 505 <= fore_finger[0] <= 600:
                        color_index = 3
                else:
                    if color_index == 0:
                        bpoints[blue_index].appendleft(fore_finger)
                    elif color_index == 1:
                        gpoints[green_index].appendleft(fore_finger)
                    elif color_index == 2:
                        rpoints[red_index].appendleft(fore_finger)
                    elif color_index == 3:
                        ypoints[yellow_index].appendleft(fore_finger)

        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paint_window, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        cv2.imshow("Paint", paint_window)

    cv2.putText(frame, f"Mode: {mode}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        running = False

cap.release()
cv2.destroyAllWindows()
