import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load trained model
model = load_model("sign_model.keras")

classes = ['0'] + [chr(ord('A') + i) for i in range(26)]

IMG_SIZE = 128

cap = cv2.VideoCapture(0)

sentence = ""

# store last predictions
prediction_buffer = deque(maxlen=15)

last_letter = ""
frame_count = 0


def preprocess_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),2)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    resized = cv2.resize(thresh,(IMG_SIZE,IMG_SIZE))

    normalized = resized/255.0

    reshaped = normalized.reshape(1,IMG_SIZE,IMG_SIZE,1)

    return reshaped, thresh


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    roi = frame[100:400,100:400]

    processed, thresh = preprocess_image(roi)

    prediction = model.predict(processed, verbose=0)

    class_index = np.argmax(prediction)

    confidence = prediction[0][class_index]

    prediction_buffer.append(class_index)

    # smoothing prediction
    if len(prediction_buffer) == prediction_buffer.maxlen:

        stable_index = max(set(prediction_buffer),
                           key=prediction_buffer.count)

        letter = classes[stable_index]

        accuracy = confidence * 100

        frame_count += 1

        # automatic letter capture
        if frame_count > 20 and letter != last_letter:

            sentence += letter
            last_letter = letter
            frame_count = 0

    else:
        letter = ""
        accuracy = 0


    # draw ROI box
    cv2.rectangle(frame,(100,100),(400,400),(0,255,0),2)

    # prediction text
    cv2.putText(frame,
                f"Prediction: {letter}",
                (100,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    # accuracy text
    cv2.putText(frame,
                f"Confidence: {accuracy:.2f}%",
                (100,450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,0,0),
                2)

    # sentence display
    cv2.putText(frame,
                f"Sentence: {sentence}",
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2)

    # UI info
    cv2.putText(frame,
                "Press Q to Quit",
                (10,470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200,200,200),
                1)

    cv2.imshow("Sign Language Recognition", frame)

    cv2.imshow("Processed Hand", thresh)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break


print("Final Sentence:", sentence)

cap.release()
cv2.destroyAllWindows()