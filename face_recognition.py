import cv2
import os
from datetime import datetime
import name_utils


def recognize_and_attendance(filename):
    face_cascade = cv2.CascadeClassifier('pretrained/haarcascade_frontalface_alt2.xml')
    trained_rec = cv2.face.LBPHFaceRecognizer_create()
    trained_rec.read("traindata/trainingLBPH.xml")
    if os.path.exists(filename + ".csv"):
        filename = filename + "(1)"
    cap = cv2.VideoCapture(0)
    file_name = filename + "_log.txt"
    # Ensure attendance log exists
    if not os.path.exists(file_name):
        open(file_name, "w").close()

    # Set to keep track of recognized people to avoid duplicate entries
    recognized_persons = set()

    while True:
        r, cur_frame = cap.read()
        grey_img = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        
        # Finds face on image with 1.3 scaleFactor and 5 minNeighbors
        detected_faces = face_cascade.detectMultiScale(grey_img, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            cropped_img = grey_img[y:y + h, x:x + w]

            try:
                person_id, confidence = trained_rec.predict(cropped_img)
                
                # Threshold for confidence value
                if confidence < 30:
                    person_name = name_utils.get_name_by_id(person_id)

                    # Check if person already recognized in this session
                    if person_name not in recognized_persons:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Log attendance
                        with open(file_name, "a") as log_file:
                            print(person_name + " was recognized at " + timestamp)
                            log_file.write(f"{person_name},{timestamp}\n")

                        recognized_persons.add(person_name)

                    # Draw rectangle and person_name
                    cv2.rectangle(cur_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(cur_frame, f"{person_name} ({confidence:.2f})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
            except Exception as e:
                print(f"Error: {e}")

        cv2.imshow('Attendance System', cur_frame)
        cv2.moveWindow('Attendance System', 500, 50)

        # Break loop if session ended (file with attendance at the created)
        if cv2.waitKey(1) & (os.path.exists(filename + ".csv")):
            break

    cap.release()
    cv2.destroyAllWindows()
