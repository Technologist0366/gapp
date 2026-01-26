import os
import cv2
import numpy as np
import re
from paddleocr import PaddleOCR

# ────────────────────────────────────────────────────────────────
# Preprocessing function (unchanged – good quality for receipts)
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
# Text cleaning function – fixes common OCR errors
# ────────────────────────────────────────────────────────────────
def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # Common OCR fixes for Kenyan receipts
    replacements = {
        r'Tweaty': 'Twenty',
        r'Jaseph|Jareph': 'Joseph',
        r'T·LIB|M-LIB': 'T. LIB',
        r'NyeriTel': 'Nyeri Tel',
        r'Emall': 'Email',
        r'infonyericouty': 'infonyericounty',
        r'STOOO': '1000',
        r'材': '',
        r'PO\.BO1101000612': 'PO.BOX 1112-10100 Nyeri Tel:0612030700',
    }
    
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    
    # Normalize spaces and currency
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bKshs?\.?\s*', 'KSh ', text, flags=re.I)
    
    return text

# ────────────────────────────────────────────────────────────────
# Structured field extraction – basic but useful rules
# ────────────────────────────────────────────────────────────────
def extract_key_fields(text_lines):
    data = {
        'receipt_number': None,
        'date': None,
        'payer': None,
        'amount_words': None,
        'total_ksh': None,
        'phone': None,
        'email': None,
    }
    
    all_text = ' '.join([str(t) for t in text_lines]).lower()
    
    # Receipt / invoice number (6+ digits)
    match = re.search(r'\b\d{6,}\b', all_text)
    if match:
        data['receipt_number'] = match.group(0)
    
    # Date
    date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', all_text)
    if date_match:
        data['date'] = date_match.group(0)
    
    # Payer / Received from
    if 'received from' in all_text:
        for line in text_lines:
            if 'received from' in str(line).lower():
                data['payer'] = str(line).strip()
                break
    
    # Amount in words (contains 'only')
    for line in text_lines:
        if 'only' in str(line).lower():
            data['amount_words'] = str(line).strip()
            break
    
    # Amount in numbers + KSh
    amount_match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(KSh|Shs)', all_text, re.I)
    if amount_match:
        data['total_ksh'] = amount_match.group(1)
    
    # Phone (Kenyan formats)
    phone_match = re.search(r'(0\d{9}|7\d{8}|\+254\d{9})', all_text)
    if phone_match:
        data['phone'] = phone_match.group(0)
    
    # Email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', all_text)
    if email_match:
        data['email'] = email_match.group(0)
    
    return data

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
results = ocr.predict(processed_path)

print("\nRaw OCR Results:")
all_texts = []
all_scores = []

for res in results:
    for text, score in zip(res.get('rec_texts', []), res.get('rec_scores', [])):
        all_texts.append(text)
        all_scores.append(score)
        conf_str = f"{score:.3f}" if score is not None else "N/A"
        print(f"  {text:50} (conf: {conf_str})")

# ── Cleaned & structured output ─────────────────────────────────
print("\nCleaned & normalized text:")
for text, score in zip(all_texts, all_scores):
    cleaned = clean_text(text)
    if cleaned:
        conf_str = f"(conf: {score:.3f})" if score is not None else "(conf: N/A)"
        print(f"  {cleaned:50} {conf_str}")

print("\nStructured extraction attempt:")
fields = extract_key_fields(all_texts)
for key, value in fields.items():
    if value:
        print(f"  {key:18}: {value}")

print("\nDone!")
