from typing import List, Tuple
import numpy as np
from retinaface import RetinaFace
import cv2

class FaceDetector:
    def __init__(self):
        # Load your model here (MTCNN, RetinaFace, etc.)
        pass

    def detect_faces(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces and return bounding boxes [(x1, y1, x2, y2), ...]
        """
        faces = RetinaFace.detect_faces(image_path)
        return faces
    
    def draw_boxes(self, image_path: str, faces):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_crops = []
        for face in faces.values():
            x1, y1, x2, y2 = face['facial_area']
            face_img = image_rgb[y1:y2, x1:x2]

            # Resize to FaceNet input size (160x160)
            face_resized = cv2.resize(face_img, (160, 160))
            face_crops.append(face_resized)