import cv2
import mediapipe as mp
import pyautogui  # Import pyautogui for mouse control


# Function to check if fist is detected
def is_fist(landmarks):
    # Assuming the fist is formed when the thumb and index fingers are tucked in (basic example)
    # Modify this condition based on your fist detection logic
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    if index_tip.y > thumb_tip.y:
        return True
    return False


# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Flip the frame for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (MediaPipe works with RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(rgb_frame)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check for fist and click mouse
                if is_fist(hand_landmarks):
                    # Display message for fist detection
                    cv2.putText(frame, "Fist Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Move the mouse to the center of the screen and click when fist is detected
                    pyautogui.click()

        # Show the frame with landmarks and message
        cv2.imshow("Webcam - Hand Landmarks", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
