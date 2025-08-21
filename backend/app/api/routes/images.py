from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import io
import json
from pathlib import Path
import pickle
from unittest import result
from app.services.face_detection import FaceDetector
from pdf2image import convert_from_path
from app.common.image_utils import save_image
import cv2
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import pymupdf
import pytesseract
from sqlalchemy.orm import Session
from typing import Any
import shutil
import os
from uuid import uuid4
from deepface import DeepFace
import numpy as np
import time
import layoutparser as lp


from app.api.deps import CurrentUser, SessionDep
from app.models import CompareImage, FaceEmbeddings, Image as ImageModel, ImagePublic, MatchImage  # Response schema
from PIL import Image, ImageDraw
from sqlmodel import select, func
from retinaface import RetinaFace
from transformers import DonutProcessor, VisionEncoderDecoderModel

router = APIRouter(prefix="/images", tags=["images"])

UPLOAD_DIR = "uploads/images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

detector = FaceDetector()

# @router.post("/", response_model=ImagePublic)
@router.post("/")
def upload_image(
    *,
    session: SessionDep,
    
    file: UploadFile = File(...)
) -> Any:
    """
    Upload an image.
    """
    start_time = time.time()
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    original = Image.open(file.file).convert("RGB")
    original.thumbnail((1024, 1024))
    original_path = save_image(image=original, subdir="original")

    print(f"upload time : {time.time() - start_time:.2f}")
    start_time = time.time()

    detections = DeepFace.extract_faces(
        img_path=original_path,
        # detector_backend='retinaface',
        # detector_backend='yunet',
        detector_backend='retinaface',
        align=True,
        enforce_detection=False
    )
    
    print(f"extraction time : {time.time() - start_time:.2f}")
    start_time = time.time()
        
    image = cv2.imread(original_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    print("DETECTIONS: ", len(detections))

    # Draw bounding boxes
    for det in detections:
        # print(det["facial_area"])
        x = det["facial_area"]["x"]
        y = det["facial_area"]["y"]
        w = det["facial_area"]["w"]
        h = det["facial_area"]["h"]
        draw.rectangle([(x, y), (x + w, y + h)], outline="white", width=2)

    # Save annotated image
    # output_path = "f1_podium_annotated.jpg"
    # pil_image.save(output_path)
    original_path = save_image(image=pil_image, subdir="annotated")
    
    print(f"draw and save bounding box time : {time.time() - start_time:.2f}")
    start_time = time.time()
    
    image = ImageModel.model_validate(
        {
            "filename": file.filename, 
            "path": original_path, 
            "owner_id": "1549464e-c328-4b84-8255-bf53f3df1790"
        }
    )
    session.add(image)
    session.commit()
    session.refresh(image)  # image.id is now available
    
    print(f"save metadata into db : {time.time() - start_time:.2f}")
    start_time = time.time()


    embeddings_model = []
    # for idx, face in enumerate(detections):
        # face_img = face['face']
        
        # cv2.imwrite(f"ext_face_output{uuid4()}.jpg", face_img)

    embeddings = DeepFace.represent(
        img_path=original_path,
        model_name='Facenet512',
        detector_backend='retinaface',
        enforce_detection=False,
    )

    for index, embedding in enumerate(embeddings):

        embedding_item = embedding["embedding"]  # Should be a list of 512 floats
        
        # check if embedding exists in the database in filter categories
        # database query to calculate nearest neighbours
        THRESHOLD = 0.6  # Adjust based on your distance metric

        count = session.exec(
            select(func.count())
            .where(FaceEmbeddings.embedding.cosine_distance(embedding_item) <= THRESHOLD)
        ).one()
        
        print("COUNT NEAR ONES")
        print(count)

        # if count == 0:
        embeddings_model.append(
            FaceEmbeddings.model_validate({
                "embedding": embedding_item,
                "image_id": image.id
            })
        )

    print(f"getting embeddings : {time.time() - start_time:.2f}")
    start_time = time.time()
    
    session.add_all(embeddings_model)
    session.commit()

    response = {
        "filename": file.filename, 
        "path": original_path, 
        "owner_id": "1549464e-c328-4b84-8255-bf53f3df1790",
        "embeddings": embeddings_model
    }

    return response

@router.post("/match")
def matching_faces(
    *,
    session: SessionDep,
    image: MatchImage
    ):
    
    # Detect and extract all the face embeddings
    # Query database to fetch similar faces 
    
    detections = DeepFace.extract_faces(
        img_path=image.image_base64,
        detector_backend='retinaface',
        align=True,
        enforce_detection=False
    )
    
    embeddings = []
    for idx, face in enumerate(detections):
        face_img = face['face']

        embedding = DeepFace.represent(
            img_path=face_img,
            model_name='Facenet512',
            enforce_detection=False
        )

        embedding_item = embedding[0]["embedding"]  # Should be a list of 512 floats
        embeddings.append(embedding_item)

    all_matches = []
    for index, embedding in enumerate(embeddings):
        start_time = time.time()
        results = session.exec(select(FaceEmbeddings.id).order_by(FaceEmbeddings.embedding.l1_distance(embeddings[index])).limit(1)).all()
        print(f"query time : {time.time() - start_time:.2f}")
        #  max_inner_product, cosine_distance, l1_distance, hamming_distance, and jaccard_distance
        for res_index, result in enumerate(results):
            all_matches.append(result)

    print(all_matches)

    return all_matches

@router.post("/compare")
def compare_faces(
    *,
    session: SessionDep,
    
    image: CompareImage
    ):

    result = DeepFace.verify(
        img1_path=image.image_base64_1, 
        img2_path=image.image_base64_2,
        model_name='Facenet512',
        detector_backend='retinaface',
        distance_metric='cosine'
    )
    
    return result

"""
convert get main face embeddings
convert pdf into images
    for every image get embeddings
compare all embeddings with main face embedding
"""
@router.post("/frs_job")
def frs_job(
    *,
    session: SessionDep,
    # 
    main_applicant: UploadFile = File(...),
    kyc_document: UploadFile = File(...),
    ):
    
    if not main_applicant.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    original_applicant = Image.open(main_applicant.file).convert("RGB")
    original_applicant_np = np.array(original_applicant)
    original_applicant_path = save_image(image=original_applicant, subdir="original")
    
    # applicant_result = DeepFace.represent(
    #     original_applicant_path,
    #     model_name="Facenet512",
    #     detector_backend="retinaface"
    # )
    
    # print(applicant_result)
    
    save_path = Path("uploads") / kyc_document.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(kyc_document.file, buffer)
        
    images = convert_from_path(save_path)

    print("EXTRACTED IMAGES")

    text_list = []
    for i, img in enumerate(images):
        
        img.thumbnail((1024, 1024))
        
        image_np = np.array(img)
        
        # gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # if(len(faces) == 0):
        #     continue
        
        print("EXTRACTED IMAGES: Each ", i)
        
        page_face_result = DeepFace.verify(
            img1_path=original_applicant_np,
            img2_path=image_np,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=False
        )
        # page_face_result = DeepFace.represent(
        #     img_path=np.array(img),
        #     model_name="Facenet512",
        #     detector_backend="retinaface",
        #     enforce_detection=False
        # )
        
        # print(page_face_result)

    return text_list

def process_page_bytes(image_data, index):
    img = Image.open(image_data).convert("RGB")
    img.thumbnail((1024, 1024))
    image_np = np.array(img)

    detections = DeepFace.extract_faces(
        img_path=image_np,
        detector_backend='mtcnn',
        align=True,
        enforce_detection=False
    )
    print(f"faces detected: {len(detections)}")
    return {"page": index, "faces": len(detections)}

def run_process_page_bytes(args):
    return process_page_bytes(*args)

def process_page_array(args):
    """
    Process a single page for face detection.
    Args is a tuple of (image_data, index) where image_data is the serialized image.
    """
    image_data, index = args
    
    # Deserialize the image data
    image = pickle.loads(image_data)
    
    # Resize image
    image.thumbnail((1024, 1024))
    image_np = np.array(image)
    
    # Face detection
    detections = DeepFace.extract_faces(
        img_path=image_np,
        detector_backend='mtcnn',
        align=True,
        enforce_detection=False
    )
    
    print(f"Page {index}: faces detected: {len(detections)}")
    return {"page": index, "faces": len(detections)}

@router.post("/v2/frs_job", name="FRS_JOB_V2 -  MULTI THREADING")
def frs_job(
    *,
    session: SessionDep,
    main_applicant: UploadFile = File(...),
    kyc_document: UploadFile = File(...),
    ):
    
    if not main_applicant.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    original_applicant = Image.open(main_applicant.file).convert("RGB")
    original_applicant_np = np.array(original_applicant)
    original_applicant_path = save_image(image=original_applicant, subdir="original")
    
    # applicant_result = DeepFace.represent(
    #     original_applicant_path,
    #     model_name="Facenet512",
    #     detector_backend="retinaface"
    # )
    
    # print(applicant_result)
    
    save_path = Path("uploads") / kyc_document.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(kyc_document.file, buffer)
        
    images = convert_from_path(save_path)

    print("EXTRACTED IMAGES")
    
    # ******************************* MULTIPLE THREADS *********************************
    # images_np = [np.array(img.resize((1024, 1024))) for img in images]
    
    def process_page_array(image, index):
        image.thumbnail((1024, 1024))
        image_np = np.array(image)

        detections = DeepFace.extract_faces(
            img_path=image_np,
            detector_backend='mtcnn',
            align=True,
            enforce_detection=False
        )
        print(f"faces detected: {len(detections)}")
        return {"page": index, "faces": len(detections)}
    
    # with ThreadPoolExecutor(max_workers=1) as executor:
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_page_array, images, range(len(images))))
    
    # ******************************* SINGLE THREAD *********************************
    # text_list = []
    # for i, img in enumerate(images):
    #     img.thumbnail((1024, 1024))
    #     image_np = np.array(img)
        
    #     print("EXTRACTED IMAGES: Each ", i)
    #     # start_time = time.time()
    #     # faces = RetinaFace.detect_faces(img_path=image_np, model=retinaface_model)
    #     # print(f"detection time: {time.time() - start_time:.2f}")
        
    #     detections = DeepFace.extract_faces(
    #         img_path=image_np,
    #         detector_backend='mtcnn',
    #         align=True,
    #         enforce_detection=False
    #     )
        
    #     print(f"no of faces detected: {len(detections)}")
        
    # ******************************* PROCESS POOL *********************************

    # start_time = time.time()
    # serialized_images = []
    # for i, image in enumerate(images):
    #     # Serialize the image data
    #     image_data = pickle.dumps(image)
    #     serialized_images.append((image_data, i))
    # print(f"pickle time {time.time() - start_time:.2f} sec")
        
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     results = list(executor.map(process_page_array, serialized_images))
    
    return "text_list"

@router.post("/v3/frs_job", name="FRS_JOB_V3 -  IMAGE DOWNLOAD FROM PDF")
def frs_job(
    *,
    session: SessionDep,
    # main_applicant: UploadFile = File(...),
    kyc_document: UploadFile = File(...),
    ):
    
    save_path = Path("uploads") / kyc_document.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(kyc_document.file, buffer)
        
    print("EXTRACTING IMAGES")
    
    doc = pymupdf.open(save_path) # open a document
    
    for i, page in enumerate(doc):
        print(f"\nPage {i + 1}:")
        
        # Text blocks
        text_blocks = page.get_text("blocks")
        print(f"  Text blocks: {len(text_blocks)}")

        # Images
        images = page.get_images(full=True)
        print(f"  Images: {len(images)}")

        # Drawings (vector graphics)
        drawings = page.get_drawings()
        print(f"  Drawings: {len(drawings)}")

        # Annotations (e.g., highlights, comments)
        annotations = list(page.annots() or [])
        print(f"  Annotations: {len(annotations)}")
        images = page.get_images()  
        print(images)
        if len(images) > 0:
            print(type(images[0]))
            
        for img_index, img in enumerate(images):
            xref = img[0]
            # pix = pymupdf.Pixmap(doc, xref)
            # if pix.n < 5:
            #     pix.save(f"page_{i}_img_{img_index}.jpg")
            # else:
            #     pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            #     pix.save(f"page_{i}_img_{img_index}.jpg")
                
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_img.save(f'image_save_pymupdf_page_{i}image{img_index}.jpg')
            pil_img_np = np.array(pil_img)
            
            detections = DeepFace.extract_faces(
                img_path=pil_img_np,
                detector_backend='mtcnn',
                align=True,
                enforce_detection=False
            )
            
            print(f" No of faces detected: {len(detections)}")
            
    
    return "text_list"

@router.post("/v4/frs_job", name="FRS_JOB_V4 -  IMAGES EXTRACT FROM PDF - OCR - OPEN CV")
def frs_job(
    *,
    session: SessionDep,
    # main_applicant: UploadFile = File(...),
    kyc_document: UploadFile = File(...),
    ):
    
    save_path = Path("uploads") / kyc_document.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(kyc_document.file, buffer)
        
    images = convert_from_path(save_path)

    print("EXTRACTED IMAGES")

    text_list = []
    for i, img in enumerate(images):
        
        img.thumbnail((1024, 1024))
        
        image_np = np.array(img)

        img_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Find contours to detect image regions
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour_index, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small regions
            if w > 100 and h > 100:
                cropped = img_cv[y:y+h, x:x+w]
                cv2.imwrite(f"opencv_extraction_page_{i}_crop_{contour_index}.png", cropped)

        # TAKDS time
        # data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Bit faster, but can use retinaface directly instead of this
        # custom_config = r'--psm 11 -c tessedit_do_invert=0'
        # boxes = pytesseract.image_to_boxes(img, output_type=pytesseract.Output.DICT, config=custom_config)


        # print("DATA: " , boxes)

        print("EXTRACTED IMAGES: Each ", i)

    return text_list

@router.post("/v5/frs_job", name="FRS_JOB_V5 -  IMAGES EXTRACT FROM PDF - USING ML/DL MODELS")
def frs_job(
    *,
    session: SessionDep,
    # main_applicant: UploadFile = File(...),
    kyc_document: UploadFile = File(...),
    ):
    
    save_path = Path("uploads") / kyc_document.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(kyc_document.file, buffer)
        
    images = convert_from_path(save_path)

    print("EXTRACTED IMAGES")

    text_list = []

    model = lp.Detectron2LayoutModel(
        # 'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        'lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        # label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
        label_map={1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"},
        # model_path='/Users/rohannaidu/.torch/iopath_cache/s/57zjbwv6gh3srry/model_final.pth'
        model_path='/Users/rohannaidu/.torch/iopath_cache/s/h7th27jfv19rxiy/model_final.pth'
        )
    
    for i, img in enumerate(images):
        image = img.convert("RGB")
        
        layout = model.detect(image)
        for block in layout:
            if block.type == "ImageRegion":
                x1, y1, x2, y2 = map(int, block.coordinates)
                img.crop((x1, y1, x2, y2)).save(f"graphic_page_{i}_crop_{x1}_{y1}.png")
        
        # layout = model.detect(image)
        
        # image_updated = lp.draw_box(image, layout, box_width=3, box_color='red')
        
        # image_updated.save(f"LP_IMG_{i}.jpg")
        
        # print(type(layout))
        
        print("EXTRACTED IMAGES: Each ", i)

    return text_list

# def frs_job(
#     *,
#     session: SessionDep,
#     # main_applicant: UploadFile = File(...),
#     kyc_document: UploadFile = File(...),
#     ):
    
#     save_path = Path("uploads") / kyc_document.filename
#     save_path.parent.mkdir(parents=True, exist_ok=True)
    
#     with open(save_path, "wb") as buffer:
#         shutil.copyfileobj(kyc_document.file, buffer)
        
#     images = convert_from_path(save_path)

#     print("EXTRACTED IMAGES")

#     text_list = []
#     # model = lp.Detectron2LayoutModel(
#     #     config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
#     #     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
#     #     label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
#     # )
    
#     hf_token = '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'
#     model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
#     processor = DonutProcessor.from_pretrained(model_id, token=hf_token)
#     model = VisionEncoderDecoderModel.from_pretrained(model_id, token=hf_token)
#     model.eval()
    
#     for i, img in enumerate(images):
#         image = img.convert("RGB")
#         # img.thumbnail((1024, 1024))
        
#         # image_np = np.array(img)
        
#         # DONUT DETECTOR
#         pixel_values = processor(image, return_tensors="pt").pixel_values

#         # Run Donut
#         task_prompt = "<doclaynet-task>"
#         decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
#         outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids)
#         result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

#         print("RESULT FROM DONUT")
#         print(type(result))
#         print(result)

#         # layout_data = json.loads(result.replace("<s_doclaynet>", "").replace("</s_doclaynet>", "").strip())

#         # Crop and save figures
#         # for j, block in enumerate(layout_data):
#         #     if block["label"].lower() == "figure":
#         #         box = block["bbox"]  # [x, y, width, height]
#         #         w, h = image.size
#         #         x1 = int(box[0] * w)
#         #         y1 = int(box[1] * h)
#         #         x2 = int((box[0] + box[2]) * w)
#         #         y2 = int((box[1] + box[3]) * h)
#         #         crop = image.crop((x1, y1, x2, y2))
#         #         crop.save(f"figures/page_{i+1}_figure_{j+1}.png")
        
#         # LAYOUT PARSER
#         # layout = model.detect(img)
#         # for block in layout:
#         #     if block.type == "Figure":
#         #         x1, y1, x2, y2 = map(int, block.coordinates)
#         #         img.crop((x1, y1, x2, y2)).save(f"graphic_crop_{x1}_{y1}.png")

        
#         print("EXTRACTED IMAGES: Each ", i)

#     return text_list
