import cv2
import numpy as np
import name_utils


def capture_faces(name):
    face_cascade = cv2.CascadeClassifier('pretrained/haarcascade_frontalface_alt2.xml')

    # Get new person person_id and name
    person_id = name_utils.add_name(name)
    photo_n = 0
    img_capt = cv2.VideoCapture(0)

    # Take 50 photos
    while photo_n < 50:
        ret, eyes_img = img_capt.read()
        grey_img = cv2.cvtColor(eyes_img, cv2.COLOR_BGR2GRAY)

        # Checking that the illumination is medium
        if np.average(grey_img) > 110:
            # Finds face on image with 1.3 scaleFactor and 5 minNeighbors
            detected_faces = face_cascade.detectMultiScale(grey_img, 1.3, 5)
            # x, y - top left corner coordinates of face and w, h - face width and height
            for (x, y, w, h) in detected_faces:
                # Crop with a little margin around the edges
                cropped_img = grey_img[y - int(h / 2): y + int(h * 1.5), x - int(x / 2): x + int(w * 1.5)]

                # Try to detect and straighten eye_list
                eyes_img = detect_eyes(cropped_img)

                if eyes_img is not None:
                    # Aligned to eye_list image
                    cur_frame = eyes_img
                else:
                    cur_frame = grey_img[y: y + h, x: x + w]

                # Save captured face
                cv2.imwrite(f"photos/User.{person_id}.{photo_n}.jpg", cur_frame)
                cv2.imshow("Captured photo", cur_frame)
                cv2.moveWindow("Captured photo", 1100, 50)
                photo_n += 1
                cv2.waitKey(300)

        cv2.imshow('Face Recognition System Capture detected_faces', grey_img)
        cv2.moveWindow('Face Recognition System Capture detected_faces', 450, 50)

    print('Face capture is complete')
    img_capt.release()
    cv2.destroyAllWindows()


def detect_eyes(image):
    eye_cascade = cv2.CascadeClassifier('pretrained/haarcascade_eye.xml')
    # Searches for eye_list in the image and returns a list of rectangles with eye coordinates
    eye_list = eye_cascade.detectMultiScale(image)

    if len(eye_list) == 2:
        # Calculate eye_angle and rotate if needed
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eye_list

        # Calculate rotation angle based on eye positions
        center_x1 = x1 + w1 // 2
        center_y1 = y1 + h1 // 2
        center_x2 = x2 + w2 // 2
        center_y2 = y2 + h2 // 2

        dx = center_x2 - center_x1
        dy = center_y2 - center_y1

        eye_angle = np.degrees(np.arctan2(dy, dx))

        # Rotate image
        height, width = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), eye_angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        return rotated_image

    return None
