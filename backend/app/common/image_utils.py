import os
from PIL import Image, ImageDraw
from uuid import uuid4
import numpy as np

BASE_UPLOAD_DIR = "uploads/images"

def save_image(image: Image.Image, subdir: str) -> str:
    os.makedirs(os.path.join(BASE_UPLOAD_DIR, subdir), exist_ok=True)
    filename = f"{uuid4().hex}.jpg"
    path = os.path.join(BASE_UPLOAD_DIR, subdir, filename)
    image.save(path)
    return path

def draw_boxes(image: Image.Image, boxes: list) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image

def pil_to_np(pil_image: Image.Image) -> np.ndarray:
    return np.array(pil_image)
