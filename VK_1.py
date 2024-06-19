import cv2
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Virtual Keyboard layout with key size and positions
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
        draw_keyboard(image, keys, None)  # Pass None for selected_key initially

        # Draw hand landmarks and interact with keyboard
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark]

                # Draw landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the index finger tip is touching any key
                for key, pos in keys:
                    x, y = pos
                    if (len(lm_list) > 8 and  # Check if enough landmarks are detected
                        point_inside_rect(lm_list[8], (x, y, key_size, key_size))):
                        # You can add key press functionality here
                        print(f"Key pressed: {key}")
                        break

        # Display the image
        cv2.namedWindow('Virtual Keyboard', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Virtual Keyboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Virtual Keyboard', image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
