import cv2
import numpy as np
import name_utils


def capture_faces(name):
    face_cascade = cv2.CascadeClassifier('pretrained/haarcascade_frontalface_alt2.xml')

    # Get new person ID and name
    ID = name_utils.add_name(name)
    Count = 0
    cap = cv2.VideoCapture(0)

    # Take 50 photos
    while Count < 50:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if np.average(gray) > 110:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                FaceImage = gray[y - int(h / 2): y + int(h * 1.5), x - int(x / 2): x + int(w * 1.5)]

                # Try to detect and straighten eyes
                Img = detect_eyes(FaceImage)

                if Img is not None:
                    frame = Img
                else:
                    frame = gray[y: y + h, x: x + w]

                # Save captured face
                cv2.imwrite(f"photos/User.{ID}.{Count}.jpg", frame)
                cv2.imshow("CAPTURED PHOTO", frame)
                Count += 1
                cv2.waitKey(300)

        cv2.imshow('Face Recognition System Capture Faces', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('FACE CAPTURE FOR THE SUBJECT IS COMPLETE')
    cap.release()
    cv2.destroyAllWindows()


def detect_eyes(image):
    eye_cascade = cv2.CascadeClassifier('pretrained/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(image)

    if len(eyes) == 2:
        # Calculate angle and rotate if needed
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes

        # Calculate rotation angle based on eye positions
        center_x1, center_y1 = x1 + w1 // 2, y1 + h1 // 2
        center_x2, center_y2 = x2 + w2 // 2, y2 + h2 // 2

        dx = center_x2 - center_x1
        dy = center_y2 - center_y1

        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate image
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))

        return rotated

    return None
