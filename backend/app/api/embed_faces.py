from app.services.face_detection import FaceDetector
from app.common.image_utils import save_image
from fastapi import Depends
from sqlalchemy.orm import Session
from deepface import DeepFace
from PIL import Image, ImageDraw
from sqlmodel import select
from app.api.deps import get_db, CurrentUser
from app.models import FaceEmbeddings, Image as ImageModel
from pathlib import Path
import cv2
import numpy as np
import uuid
import time

start_time = time.time()
# Setup
folder = Path("/Users/rohannaidu/Documents/dataset/WIDER_val/images/0--Parade")
output_folder = folder / "annotated"
output_folder.mkdir(exist_ok=True)

# Dependencies
session: Session = next(get_db())
# current_user = CurrentUser()  # Replace with actual user in your app

for file_no, file in enumerate(folder.glob("*.jpg"), start=1):
    print("processing file:", file_no)
    
    if not file.is_file():
        continue

    # Face detection
    detections = DeepFace.extract_faces(
        img_path=str(file),
        detector_backend='retinaface',
        align=True,
        enforce_detection=False
    )

    # Annotate image
    image = cv2.imread(str(file))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    for det in detections:
        x = det["facial_area"]["x"]
        y = det["facial_area"]["y"]
        w = det["facial_area"]["w"]
        h = det["facial_area"]["h"]
        draw.rectangle([(x, y), (x + w, y + h)], outline="white", width=2)

    annotated_path = output_folder / f"{file.stem}_annotated.jpg"
    pil_image.save(annotated_path)

    # Save image record
    image_record = ImageModel.model_validate({
        "filename": file.name,
        "path": str(file),
        "owner_id": "1549464e-c328-4b84-8255-bf53f3df1790"
    })

    session.add(image_record)
    session.commit()
    session.refresh(image_record)

    # Save embeddings
    embeddings_model = []
    for face in detections:
        face_img = face["face"]

        embedding = DeepFace.represent(
            img_path=face_img,
            model_name='Facenet512',
            enforce_detection=False
        )

        embedding_item = embedding[0]["embedding"]
        embeddings_model.append(
            FaceEmbeddings.model_validate({
                "embedding": embedding_item,
                "image_id": image_record.id
            })
        )

    session.add_all(embeddings_model)
    session.commit()

print(f"\n⏱️ total time: {time.time() - start_time:.2f} sec")