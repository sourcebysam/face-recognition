import face_recognition
import cv2
import os
import numpy as np

def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names