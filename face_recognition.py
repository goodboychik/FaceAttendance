import cv2
import os
from datetime import datetime
import name_utils


def recognize_and_attendance(filename):
    face_cascade = cv2.CascadeClassifier('pretrained/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("traindata/trainingLBPH.xml")
    if os.path.exists(filename + ".csv"):
        filename = filename + "(1)"
    cap = cv2.VideoCapture(0)
    file_name = filename + "_log.txt"
    # Ensure attendance log exists
    if not os.path.exists(file_name):
        open(file_name, "w").close()

    # Set to keep track of recognized people to avoid duplicate entries
    recognized_people = set()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            try:
                ID, confidence = recognizer.predict(face_roi)

                if confidence < 100:  # Adjust threshold as needed
                    NAME = name_utils.get_name_by_id(ID)

                    # Check if person already recognized in this session
                    if NAME not in recognized_people:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Log attendance
                        with open(file_name, "a") as log_file:
                            print(NAME + "was recognized at " + timestamp)
                            log_file.write(f"{NAME},{timestamp}\n")

                        recognized_people.add(NAME)

                    # Draw rectangle and name
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{NAME} ({confidence:.2f})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
            except Exception as e:
                print(f"Recognition error: {e}")

        cv2.imshow('Attendance System', frame)

        # Break loop if session ended (file with attendance at the created)
        if cv2.waitKey(1) & (os.path.exists(filename + ".csv")):
            break

    cap.release()
    cv2.destroyAllWindows()
