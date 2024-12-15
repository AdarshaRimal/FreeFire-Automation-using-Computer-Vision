from collections import deque
import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

landmark_history = deque(maxlen=20)  # Increased buffer size

def detect_swipe():
    if len(landmark_history) < 2:
        return None

    start_x = landmark_history[0].x
    end_x = landmark_history[-1].x
    start_y = landmark_history[0].y
    end_y = landmark_history[-1].y

    # Print debug data
    print(f"Start_x: {start_x}, End_x: {end_x}, Start_y: {start_y}, End_y: {end_y}")

    if abs(start_y - end_y) < 0.05:  # Minimal vertical movement
        if end_x - start_x > 0.1:
            return "right"
        elif start_x - end_x > 0.1:
            return "left"
    return None

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                landmark_history.append(wrist)

                gesture = detect_swipe()
                if gesture == "right":
                    cv2.putText(frame, "Swipe Right Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    pyautogui.hotkey('ctrl', 'right')
                elif gesture == "left":
                    cv2.putText(frame, "Swipe Left Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    pyautogui.hotkey('ctrl', 'left')

        cv2.imshow("Webcam - Hand Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
