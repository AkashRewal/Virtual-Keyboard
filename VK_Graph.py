import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict

# Initialize MediaPipe handsq
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Virtual Keyboard layout with slightly increased key size
key_size = 70
keys = [
    ("Q", (50, 100)), ("W", (150, 100)), ("E", (250, 100)), ("R", (350, 100)), 
    ("T", (450, 100)), ("Y", (550, 100)), ("U", (650, 100)), ("I", (750, 100)), 
    ("O", (850, 100)), ("P", (950, 100)),
    ("A", (50, 200)), ("S", (150, 200)), ("D", (250, 200)), ("F", (350, 200)), 
    ("G", (450, 200)), ("H", (550, 200)), ("J", (650, 200)), ("K", (750, 200)), 
    ("L", (850, 200)), ("Backspace", (950, 200)),
    ("Z", (50, 300)), ("X", (150, 300)), ("C", (250, 300)), ("V", (350, 300)), 
    ("B", (450, 300)), ("N", (550, 300)), ("M", (650, 300))
]

# Display text position and content
selected_keys = ""
display_pos = (50, 50)
key_pressed = False
selected_key = None
prev_selected_key = None
last_selected_key = None
last_key_time = 0

# Dictionary to track key presses
key_count = defaultdict(int)

# Function to draw keyboard
def draw_keyboard(image, keys, selected_key):
    for key, pos in keys:
        x, y = pos
        if key == selected_key:
            cv2.rectangle(image, (x, y), (x + key_size, y + key_size), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, key, (x + 15, y + 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x + key_size, y + key_size), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(image, (x, y), (x + key_size, y + key_size), (255, 0, 0), 2)
            cv2.putText(image, key, (x + 15, y + 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

# Function to check if a point is inside a rectangle
def point_inside_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx < x < rx + rw and ry < y < ry + rh

# Main loop
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        results = hands.process(image_rgb)

        # Draw the keyboard
        draw_keyboard(image, keys, selected_key)

        # Draw hand landmarks and interact with keyboard
        if results.multi_hand_landmarks:
            fingers_touching = False
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark]

                # Draw landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the index finger tip and middle finger tip are touching any key
                for key, pos in keys:
                    x, y = pos
                    if (len(lm_list) > 12 and
                        point_inside_rect(lm_list[8], (x, y, key_size, key_size)) and
                        point_inside_rect(lm_list[12], (x, y, key_size, key_size))):
                        if not key_pressed:
                            if key == last_selected_key and (cv2.getTickCount() - last_key_time) / cv2.getTickFrequency() < 1.0:
                                selected_keys = selected_keys[:-1]  # Remove last character if same key is selected within 1 second
                            else:
                                if key == "Backspace":
                                    selected_keys = selected_keys[:-1]
                                else:
                                    selected_keys += key
                                    key_count[key] += 1  # Count the key press
                                last_selected_key = key
                                last_key_time = cv2.getTickCount()
                            key_pressed = True
                            selected_key = key
                            cv2.putText(image, f"Selected Key: {key}", display_pos, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                        fingers_touching = True
                        break
                else:
                    key_pressed = False

            # Reset selected_key if fingers are not touching any key
            if not fingers_touching:
                selected_key = None

        # Display the selected keys on the screen
        cv2.putText(image, selected_keys, (50, 450), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Display the image
        cv2.namedWindow('Virtual Keyboard', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Virtual Keyboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Virtual Keyboard', image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Plotting the graph for key presses
import matplotlib.pyplot as plt

keys = list(key_count.keys())
counts = list(key_count.values())

plt.bar(keys, counts)
plt.xlabel('Keys')
plt.ylabel('Count')
plt.title('Key Press Frequency')
plt.show()

cap.release()
cv2.destroyAllWindows()
