import cv2
import mediapipe as mp
import autopy

# Initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize autopy for mouse control
screen_width, screen_height = autopy.screen.size()

# Function to map values from one range to another
def map_value(value, left_min, left_max, right_min, right_max):
    left_span = left_max - left_min
    right_span = right_max - right_min

    value_scaled = float(value - left_min) / float(left_span)

    return right_min + (value_scaled * right_span)

# Function to move the mouse cursor
def move_mouse(x, y):
    autopy.mouse.move(map_value(x, 0, screen_width, 0, screen_width),
                      map_value(y, 0, screen_height, 0, screen_height))

# Main function
def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip the image horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Set flag to False by default
            flag = False

            # Process the frame
            results = hands.process(rgb_frame)

            # If hand is detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get the position of index finger
                    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])

                    # Move the mouse cursor based on the position of the index finger
                    move_mouse(x, y)
                    flag = True

            # If no hand is detected, move the mouse to the center of the screen
            if not flag:
                move_mouse(screen_width // 2, screen_height // 2)

            # Display the frame
            cv2.imshow('AI Virtual Mouse', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

