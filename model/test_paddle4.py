import os
import cv2
import numpy as np
import re
import glob
import pandas as pd
from tqdm import tqdm
from paddleocr import PaddleOCR

# ────────────────────────────────────────────────────────────────
# Preprocessing function
# ────────────────────────────────────────────────────────────────
def preprocess_receipt(image_path, output_dir="preprocessed", skip=False):
    if skip:
        return image_path
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_processed.jpg")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read {image_path}")
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
            print(f"Preprocessed: {output_path}")
            return output_path
        else:
            return None
            
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return None

# ────────────────────────────────────────────────────────────────
# Detect issuer from raw text
# ────────────────────────────────────────────────────────────────
def detect_issuer(all_texts):
    all_text = ' '.join(all_texts).lower()
    if 'munch' in all_text or 'lavington' in all_text:
        return 'Munch Lavington'
    if 'nyeri' in all_text or 'county treasury' in all_text:
        return 'Nyeri County'
    if 'naivas' in all_text or 'supermarket' in all_text:
        return 'Naivas / Supermarket'
    if 'equity' in all_text:
        return 'Equity Bank'
    if 'shell' in all_text or 'total' in all_text and 'fuel' in all_text:
        return 'Fuel Station'
    return 'Unknown / Other'

# ────────────────────────────────────────────────────────────────
# Clean text – expanded rules for different issuers
# ────────────────────────────────────────────────────────────────
def clean_text(text, issuer='Unknown'):
    if not text or not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # General fixes
    replacements = {
        r'Tweaty|Tweenty|Twanti': 'Twenty',
        r'Jaseph|Jareph|Jaspeh|Josep': 'Joseph',
        r'T·LIB|M-LIB|T LIB': 'T. LIB',
        r'Emall|Emal|Emali': 'Email',
        r'STOOO|STOOO|1OOO|IOOO': '1000',
        r'NyeriTel|Nyerl': 'Nyeri Tel',
        r'infonyericouty|infonyeri county': 'infonyericounty',
        r'材|[\u4e00-\u9fff]': '',
        r'Recipt|Recipt|Recpt': 'Receipt',
        r'Amout|Amnt': 'Amount',
        r'P\.O\.Box|PO\.BOX|PO BOX': 'P.O. Box',
        r'VAT\s*(\d+)': r'VAT \1',
        r'Tt1 Due|Tt1|Due': 'Total Due',
        r'CR/DR CARD': 'Card Payment',
        r'CASH SALE': 'Cash Sale',
        r'Pan seared': 'Pan-seared',
        r'Ceak': 'Steak',
        r'Bot.of': 'Bottle of',
        r'KARTASI': 'Kartasi',  # assuming brand name
    }
    
    # Issuer-specific
    if issuer == 'Munch Lavington':
        replacements.update({
            r'BURUBURU BRANCH': 'Buruburu Branch',
            r'P.0. Box': 'P.O. Box',
            r'Vi1lage Market': 'Village Market',
            r'MUNCH LAVINGTON': 'Munch Lavington',
            r'Cts.': 'Cts.',  # assuming abbreviation
            r'JQSO': 'JQSO',  # unknown – keep or correct if known
        })
    elif issuer == 'Nyeri County':
        replacements.update({
            r'PO\.BO1101000612': 'P.O. Box 1112-10100 Nyeri Tel:0612030700',
        })
    
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(Kshs?|Shs|KSHS)\.?\s*', 'KSh ', text, flags=re.I)
    text = re.sub(r'[-–—]\s*', '-', text)
    
    return text

# ────────────────────────────────────────────────────────────────
# Extract key fields – improved
# ────────────────────────────────────────────────────────────────
def extract_key_fields(text_lines, issuer='Unknown'):
    data = {
        'issuer': issuer,
        'receipt_number': None,
        'date': None,
        'payer': None,
        'amount_words': None,
        'total_ksh': None,
        'phone': None,
        'email': None,
        'subtotal': None,
        'tax': None,
        'branch': None,
    }
    
    all_text = ' '.join(text_lines).lower()
    
    # General
    data['receipt_number'] = re.search(r'\b\d{6,}\b', all_text).group(0) if re.search(r'\b\d{6,}\b', all_text) else None
    data['date'] = re.search(r'\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\'?\d{2,4}', all_text, re.I).group(0) if re.search(r'\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\'?\d{2,4}', all_text, re.I) else None
    data['total_ksh'] = re.search(r'(subtotal|total|due)\s*kes?\s*(\d+[,.\d]*)', all_text, re.I).group(2) if re.search(r'(subtotal|total|due)\s*kes?\s*(\d+[,.\d]*)', all_text, re.I) else None
    
    # Issuer-specific
    if issuer == 'Munch Lavington':
        data['branch'] = re.search(r'(\w+ branch)', all_text, re.I).group(1) if re.search(r'(\w+ branch)', all_text, re.I) else None
        data['date'] = re.search(r'\d{2}:\d{2}\s*(am|pm)\s*mon|tue|wed|thu|fri|sat|sun\s*\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\'?\d{2,4}', all_text, re.I).group(0) if re.search(r'\d{2}:\d{2}\s*(am|pm)\s*mon|tue|wed|thu|fri|sat|sun\s*\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\'?\d{2,4}', all_text, re.I) else None
        data['payer'] = re.search(r'cashier\s*([\w\s]+)', all_text, re.I).group(1).strip() if re.search(r'cashier\s*([\w\s]+)', all_text, re.I) else None
    
    return data

# ────────────────────────────────────────────────────────────────
# Main batch processing
# ────────────────────────────────────────────────────────────────
print("Initializing PaddleOCR...")
ocr = PaddleOCR(
    lang='en',
    text_detection_model_name='PP-OCRv4_mobile_det',
    text_recognition_model_name='PP-OCRv4_mobile_rec',
    use_textline_orientation=True,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)
print("OCR engine ready!\n")

# ── Folder with receipts ────────────────────────────────────────
receipt_folder = '/home/mehit/Documents/complianceassets/model/dataset/kdreceipts/'
output_csv = 'receipts_evidence.csv'

image_paths = sorted(glob.glob(os.path.join(receipt_folder, '*.[jJ][pP][gG]')) +
                     glob.glob(os.path.join(receipt_folder, '*.[jJ][pP][eE][gG]')) +
                     glob.glob(os.path.join(receipt_folder, '*.[pP][nN][gG]')))

print(f"Found {len(image_paths)} images")

results_list = []

for img_path in tqdm(image_paths, desc="Processing receipts"):
    filename = os.path.basename(img_path)
    
    try:
        processed_path = preprocess_receipt(img_path)
        if processed_path is None:
            processed_path = img_path
        
        results = ocr.predict(processed_path)
        
        all_texts = []
        all_scores = []
        for res in results:
            all_texts.extend(res.get('rec_texts', []))
            all_scores.extend(res.get('rec_scores', []))
        
        issuer = detect_issuer(all_texts)
        cleaned_texts = [clean_text(t, issuer) for t in all_texts]
        fields = extract_key_fields(all_texts, issuer)
        
        row = {
            'filename': filename,
            'issuer': issuer,
            'raw_text': ' | '.join(all_texts),
            'cleaned_text': ' | '.join(cleaned_texts),
            'receipt_number': fields['receipt_number'],
            'date': fields['date'],
            'payer': fields['payer'],
            'amount_words': fields['amount_words'],
            'total_ksh': fields['total_ksh'],
            'phone': fields['phone'],
            'email': fields['email'],
            'num_lines': len(all_texts),
            'avg_confidence': round(sum(s for s in all_scores if s is not None) / len([s for s in all_scores if s is not None]) if any(all_scores) else 0.0, 3),
        }
        results_list.append(row)
        
        print(f"Processed {filename} - {len(all_texts)} lines - Issuer: {issuer}")
        
    except Exception as e:
        print(f"Error on {filename}: {e}")
        results_list.append({'filename': filename, 'error': str(e)})

# Save to CSV
df = pd.DataFrame(results_list)
df.to_csv(output_csv, index=False)
print(f"\nSaved {len(results_list)} receipts → {output_csv}")
print("Done!")
