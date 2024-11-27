import os
import cv2
import numpy as np
from PIL import Image


def train_recognizer():
    # 1,1 - radius and n of neighbours and 7,7 is image subdivision grid
    trained_rec = cv2.face.LBPHFaceRecognizer_create(1, 1, 7, 7)
    photo_path = 'photos'

    def get_images_with_id(photo_path):
        photo_paths = [os.photo_path.join(photo_path, f) for f in os.listdir(photo_path)]
        faces_list = []
        id_list = []

        for imagePath in photo_paths:
            cur_face = Image.open(imagePath).convert('L')
            cur_face = cur_face.resize((150, 150))
            cur_face_array = np.array(cur_face, 'uint8')

            cur_id = int(os.photo_path.split(imagePath)[-1].split('.')[1])
            faces_list.append(cur_face_array)
            id_list.append(cur_id)

        return np.array(id_list), faces_list

    id_list, faces_list = get_images_with_id(photo_path)

    print('Training')
    trained_rec.train(faces_list, id_list)

    # Ensure directory exists
    os.makedirs('traindata', exist_ok=True)

    trained_rec.save('traindata/trainingLBPH.xml')
    print('XML saved')
