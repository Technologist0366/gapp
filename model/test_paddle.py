# test_paddle.py - Improved version for better receipt OCR accuracy
import os
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import glob
import re
from datetime import datetime

# ────────────────────────────────────────────────────────────────
# Environment settings for stability on CPU (important on Kali)
# ────────────────────────────────────────────────────────────────
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['FLAGS_use_mkldnn'] = '0'          # avoid oneDNN crashes
os.environ['OMP_NUM_THREADS'] = '1'           # limit threads
os.environ['MKL_NUM_THREADS'] = '1'           # extra safety

# ────────────────────────────────────────────────────────────────
# Simple but effective pre-processing for receipts
# ────────────────────────────────────────────────────────────────
def preprocess_receipt(image_path, output_path="preprocessed.jpg"):
    """
    Improve image quality: contrast, denoising, mild sharpening, light deskew
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise + preserve edges
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Enhance contrast (CLAHE - very good for receipts)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    # Mild sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Very light deskew (simple rotation correction)
    coords = np.column_stack(np.where(sharpened > 30))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:  # only correct if meaningful rotation
            (h, w) = sharpened.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            sharpened = cv2.warpAffine(sharpened, M, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)

    # Save processed image
    cv2.imwrite(output_path, sharpened)
    print(f"Preprocessed image saved: {output_path}")
    return output_path

# ────────────────────────────────────────────────────────────────
# Clean and normalize text (fix common OCR mistakes)
# ────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.strip()
    # Common fixes for Kenyan receipts
    replacements = {
        r'Tweaty': 'Twenty',
        r'Jaseph': 'Joseph',
        r'T·LIB': 'T. LIB',
        r'NyeriTel': 'Nyeri Tel',
        r'Emall': 'Email',
        r'infonyericouty': 'infonyericounty',
        r'STOOO': '1000',          # example - adjust based on your images
        r'材': '',                 # remove stray characters
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Normalize currency / amounts
    text = re.sub(r'\bKshs?\.?\s*', 'KSh ', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text)
    return text

# ────────────────────────────────────────────────────────────────
# Extract structured fields (basic rule-based)
# ────────────────────────────────────────────────────────────────
def extract_key_fields(text_lines):
    data = {
        'receipt_number': None,
        'date': None,
        'payer': None,
        'amount_words': None,
        'amount_numbers': None,
        'total_ksh': None,
        'phone': None,
        'email': None,
    }

    for line in text_lines:
        clean_line = clean_text(line)
        lower = clean_line.lower()

        # Receipt / invoice number
        if re.search(r'\b\d{6,}\b', clean_line):
            if not data['receipt_number']:
                data['receipt_number'] = clean_line

        # Date
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', clean_line) or 'date' in lower:
            data['date'] = clean_line

        # Received from / payer
        if 'received from' in lower or 'from' in lower:
            data['payer'] = clean_line.replace('Received from', '').strip()

        # Amount in words
        if any(word in lower for word in ['only', 'shillings', 'twenty', 'thirty']):
            data['amount_words'] = clean_line

        # Amount in numbers
        match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', clean_line)
        if match and ('ksh' in lower or 'shs' in lower or 'total' in lower):
            data['total_ksh'] = match.group(1)

        # Phone
        if re.search(r'0\d{9}|7\d{8}', clean_line):
            data['phone'] = clean_line

        # Email
        if '@' in clean_line:
            data['email'] = clean_line

    return data

# ────────────────────────────────────────────────────────────────
# Main execution
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

# Pre-process for better accuracy
processed_path = preprocess_receipt(img_path)

print(f"Running OCR on processed image: {processed_path}")
results = ocr.predict(processed_path)

print("\nRaw OCR Results:")
all_texts = []
for res in results:
    res.print()                     # original output
    all_texts.extend(res['rec_texts'])

    # Save visualization
    res.save_to_img('ocr_output')

# ── Cleaned & structured output ─────────────────────────────────
print("\nCleaned & normalized text:")
for text in all_texts:
    cleaned = clean_text(text)
    if cleaned:
        print(f"  {cleaned}")

print("\nStructured extraction attempt:")
fields = extract_key_fields(all_texts)
for key, value in fields.items():
    if value:
        print(f"  {key:18}: {value}")

print("\nDone!")
