import os
import cv2
import numpy as np
from PIL import Image


def train_recognizer():
    LBPHFace = cv2.face.LBPHFaceRecognizer_create(1, 1, 7, 7)
    path = 'photos'

    def get_images_with_id(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        FaceList = []
        IDs = []

        for imagePath in imagePaths:
            faceImage = Image.open(imagePath).convert('L')
            faceImage = faceImage.resize((150, 150))
            faceNP = np.array(faceImage, 'uint8')

            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            FaceList.append(faceNP)
            IDs.append(ID)

        return np.array(IDs), FaceList

    IDs, FaceList = get_images_with_id(path)

    print('Training')
    LBPHFace.train(FaceList, IDs)

    # Ensure directory exists
    os.makedirs('traindata', exist_ok=True)

    LBPHFace.save('traindata/trainingLBPH.xml')
    print('XML saved')
