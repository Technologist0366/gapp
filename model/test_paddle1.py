import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import draw_ocr
from PIL import Image
import re
import matplotlib.pyplot as plt  # optional, but good for future use

# ────────────────────────────────────────────────────────────────
# Preprocessing function (your current version – unchanged)
# ────────────────────────────────────────────────────────────────
def preprocess_receipt(image_path, output_dir="preprocessed"):
    """
    Improve image quality: contrast, denoising, mild sharpening, light deskew
    Returns path to preprocessed image or None if failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_processed.jpg")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        coords = np.column_stack(np.where(sharpened > 30))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:
                (h, w) = sharpened.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                sharpened = cv2.warpAffine(sharpened, M, (w, h),
                                           flags=cv2.INTER_CUBIC,
                                           borderMode=cv2.BORDER_REPLICATE)
        
        success = cv2.imwrite(output_path, sharpened)
        if success:
            print(f"Preprocessed image saved: {output_path}")
            return output_path
        else:
            print(f"Failed to save preprocessed image: {output_path}")
            return None
            
    except Exception as e:
        print(f"Preprocessing error on {image_path}: {e}")
        return None

# ────────────────────────────────────────────────────────────────
# Main script
# ────────────────────────────────────────────────────────────────
print("Initializing PaddleOCR (PP-OCRv4 mobile - English)...")
ocr = PaddleOCR(
    lang='en',
    text_detection_model_name='PP-OCRv4_mobile_det',
    text_recognition_model_name='PP-OCRv4_mobile_rec',
    use_textline_orientation=True,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)
print("OCR engine ready!\n")

# ── Single image test ───────────────────────────────────────────
img_path = '/home/mehit/Documents/complianceassets/model/dataset/kdreceipts/16000-receipt.jpeg'

print(f"Preprocessing: {img_path}")
processed_path = preprocess_receipt(img_path)

if processed_path is None:
    print("Preprocessing failed → using original image")
    processed_path = img_path

print(f"Running OCR on: {processed_path}")
results = ocr.ocr(processed_path, cls=True)  # correct for PaddleOCR 2.x

print("\nRaw OCR Results:")
all_texts = []
all_scores = []

# Handle results format in 2.x
detections = results[0] if isinstance(results, list) and len(results) > 0 else results

for line in detections:
    box = line[0]
    text, score = line[1]
    all_texts.append(text)
    all_scores.append(score)
    print(f"  {text:50} (conf: {score:.3f})   box: {box}")

# ── Visualization with font fallback ─────────────────────────────
print("\nGenerating annotated image...")
image = Image.open(processed_path).convert('RGB')

boxes = [line[0] for line in detections]
texts = [line[1][0] for line in detections]
scores = [line[1][1] for line in detections]

# Try multiple font paths (one will work on Kali)
font_paths = [
    './doc/fonts/simfang.ttf',                        # PaddleOCR default
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', # Common on Debian/Kali
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
]

font_path = None
for path in font_paths:
    if os.path.exists(path):
        font_path = path
        break

if font_path:
    print(f"Using font: {font_path}")
    im_show = draw_ocr(image, boxes, texts, scores, font_path=font_path)
else:
    print("Warning: No font found – saving without text labels")
    im_show = draw_ocr(image, boxes, texts, scores, font_path=None)

im_show = Image.fromarray(im_show)
annotated_path = 'ocr_output/16000-receipt_annotated.jpg'
os.makedirs('ocr_output', exist_ok=True)
im_show.save(annotated_path)
print(f"Annotated image saved: {annotated_path}")

# ── Cleaned & structured output ─────────────────────────────────
print("\nCleaned & normalized text:")
for text, score in zip(all_texts, all_scores):
    cleaned = clean_text(text)  # assuming clean_text is defined
    if cleaned:
        print(f"  {cleaned:50} (conf: {score:.3f})")

print("\nStructured extraction attempt:")
fields = extract_key_fields(all_texts)  # assuming extract_key_fields is defined
for key, value in fields.items():
    if value:
        print(f"  {key:18}: {value}")

print("\nDone!")
