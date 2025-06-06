import cv2
import mediapipe as mp
import random
import time
import zipfile
import os

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Define Rock, Paper, Scissors gestures
GESTURES = {
    0: "Rock",    # Closed fist
    1: "Paper",   # Open hand
    2: "Scissors" # Two fingers (index and middle)
}

# Function to detect hand gesture
def detect_gesture(hand_landmarks):
    # Get the coordinates of the thumb, index, and middle finger tips
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Calculate distances between finger tips for gesture recognition
    distance_index_middle = abs(index_tip.y - middle_tip.y)  # Vertical distance between index and middle
    distance_thumb_index = abs(thumb_tip.y - index_tip.y)  # Vertical distance between thumb and index

    # Define thresholds for gesture recognition
    if distance_index_middle < 0.05 and distance_thumb_index < 0.05:
        return 0  # Rock (closed fist)
    elif distance_index_middle > 0.15 and distance_thumb_index > 0.15:
        return 1  # Paper (open hand)
    elif distance_index_middle > 0.05 and distance_thumb_index < 0.05:
        return 2  # Scissors (index and middle extended)

    return -1  # Undefined gesture

# Main function to play the game
def play_game():
    cap = cv2.VideoCapture(0)  # Use webcam for input
    gesture_detected = False  # To ensure we detect the gesture only once per round
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)
        
        # Convert frame to RGB (MediaPipe requires RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        # Initialize gesture variable
        gesture = -1
        
        # If hand landmarks are found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections on the hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Detect gesture only if it hasn't been detected yet
                if not gesture_detected:
                    gesture = detect_gesture(hand_landmarks)
                    
                    # If gesture is valid, print it and break out of loop
                    if gesture != -1:
                        cv2.putText(frame, f'Your Gesture: {GESTURES[gesture]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        gesture_detected = True  # Mark gesture as detected
                        break
        
        # Display the game frame
        cv2.imshow('Rock Paper Scissors - Hand Gesture', frame)
        # Capture a key press for making a move
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break

        if gesture_detected:  # When a gesture is detected, play the game
            # Computer's random choice
            computer_choice = random.choice([0, 1, 2])
            user_choice = gesture
            print(f'Your choice: {GESTURES[user_choice]}')
            print(f"Computer's choice: {GESTURES[computer_choice]}")

            # Determine the winner
            if user_choice == computer_choice:
                print("It's a draw!")
            elif (user_choice == 0 and computer_choice == 2) or (user_choice == 1 and computer_choice == 0) or (user_choice == 2 and computer_choice == 1):
                print("You win!")
            else:
                print("Computer wins!")

            # Wait for a few seconds before resetting for the next round
            time.sleep(2)

            # Reset for the next round
            gesture_detected = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_game()
