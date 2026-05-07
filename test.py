import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("sign_model.keras")

# Define classes (A-Z + blank)
classes = ['0'] + [chr(ord('A') + i) for i in range(26)]

IMG_SIZE = 128

# Image preprocessing function
def preprocess_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 2)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    resized = cv2.resize(thresh,(IMG_SIZE,IMG_SIZE))

    normalized = resized / 255.0

    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return reshaped


# Start webcam
cap = cv2.VideoCapture(0)

sentence = ""

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to grab frame")
        break

    # ROI box
    roi = frame[100:400,100:400]

    processed = preprocess_image(roi)

    prediction = model.predict(processed, verbose=0)

    class_index = np.argmax(prediction)

    confidence = prediction[0][class_index]

    letter = classes[class_index]

    accuracy = confidence * 100

    # Draw box
    cv2.rectangle(frame,(100,100),(400,400),(0,255,0),2)

    # Show prediction
    cv2.putText(frame,"Prediction: "+letter,
                (100,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.putText(frame,f"Accuracy: {accuracy:.2f}%",
                (100,450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(255,0,0),2)

    cv2.putText(frame,"Sentence: "+sentence,
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(255,255,255),2)

    cv2.imshow("Sign Detection",frame)

    key = cv2.waitKey(1) & 0xFF

    # Capture letter
    if key == ord('c'):
        sentence += letter

    # Space
    elif key == ord('s'):
        sentence += " "

    # Delete
    elif key == ord('d'):
        sentence = sentence[:-1]

    # Quit
    elif key == ord('q'):
        break


print("Final Sentence:", sentence)

cap.release()
cv2.destroyAllWindows()

accuracy = confidence * 100

cv2.putText(frame,f"Accuracy: {accuracy:.2f}%",
            (100,450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2)