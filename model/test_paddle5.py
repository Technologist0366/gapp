import os
import cv2
import numpy as np
import re
import glob
import pandas as pd
from tqdm import tqdm
from paddleocr import PaddleOCR

# ────────────────────────────────────────────────────────────────
# Preprocessing function (improves OCR quality)
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
            print(f"Failed to save preprocessed image: {output_path}")
            return None
            
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return None

# ────────────────────────────────────────────────────────────────
# Detect issuer (helps tailor cleaning & extraction)
# ────────────────────────────────────────────────────────────────
def detect_issuer(all_texts):
    all_text = ' '.join(all_texts).lower()
    if 'nyeri' in all_text or 'county treasury' in all_text:
        return 'Nyeri County'
    elif 'naivas' in all_text or 'supermarket' in all_text or 'carrefour' in all_text:
        return 'Supermarket'
    elif 'equity' in all_text or 'bank' in all_text:
        return 'Equity Bank'
    elif 'shell' in all_text or 'total' in all_text and 'fuel' in all_text:
        return 'Fuel Station'
    elif 'munch' in all_text or 'lavington' in all_text:
        return 'Munch Lavington'
    return 'Unknown'

# ────────────────────────────────────────────────────────────────
# Clean text – expanded rules
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
        r'材|[\u4e00-\u9fff]|行|网|！|中|专|业|厦|人': '',  # remove stray Asian chars
        r'Recipt|Recipt|Recpt': 'Receipt',
        r'Amout|Amnt': 'Amount',
        r'P\.O\.Box|PO\.BOX|PO BOX|P.0. Box': 'P.O. Box',
        r'VAT\s*(\d+)': r'VAT \1',
        r'Tt1 Due|Tt1|Due': 'Total Due',
        r'CR/DR CARD': 'Card Payment',
        r'CASH SALE': 'Cash Sale',
        r'Pan seared': 'Pan-seared',
        r'Ceak': 'Steak',
        r'Bot.of': 'Bottle of',
        r'Apri1': 'April',
        r'Vi1lage': 'Village',
        r'MpesaPaybi11': 'Mpesa Paybill',
        r'Tbone': 'T-bone',
        r'Tlapia': 'Tilapia',
        r'ugali': 'Ugali',
        r'hll oater': 'Hill Water',
        r'Q0': '0',
        r'8S0': '80',
        r'24p0': '240',
        r'Goods once sod are notReumablaThankyou': 'Goods once sold are not returnable. Thank you!',
        r'NO': 'No',
        r'DDASHARA HESTLANIS': 'Dashara Restaurants',
        r'Uat#': 'VAT#',
        r'ASHARA': 'Ashara',
        r'x*EAT-INxo#': 'Eat-In',
        r'Tex JnvNo': 'Tax Inv No',
        r'Bovars': 'Bovars',
        r'STAFF': 'Staff',
        r'Peni-Reri': 'Peni-Reri',
        r'IUenllla Hlkshake': 'Vanilla Milkshake',
        r'Tsebhl ASa': 'Tsebhl Asa',
        r'ISprlte': 'Sprite',
        r'Qutout Tax': 'Output Tax',
        r'KESS': 'KES',
        r'BAAP Polnt ot Sale': 'BAAP Point of Sale',
        r'CHICKSALESRECEIPT': 'Chick Sales Receipt',
        r'M/s.': 'M/s.',
        r'homou...': 'homou...',
        r'QTY': 'Qty',
        r'DESCRIPTION': 'Description',
        r'SHS': 'KSh',
        r'Boludayo1d': 'Boludayo1d',
        r'D280': 'D280',
        r'125400': '125400',
        r'J': 'J',
        r'0': '0',
        r'20': '20',
        r'C': 'C',
        r'L': 'L',
        r'TERMSSTRICTLYOHCASRBASIS': 'TERMS STRICTLY CASH BASIS',
        r'(50%Pro Paymt(Only s0% Aocoptad)': '(50% Pro Paymt (Only 50% Accepted))',
        r'1.whifto': '1. Whifto',
        r'2.Pk3.GrnKE': '2. Pk3 GrnKE',
        r'START UFLECLRECEIPT #': 'START OF LEGAL RECEIPT #',
        r'REGISIERH0': 'REGISTER NO',
        r'SAMRAT SUPERMARKET': 'Samrat Supermarket',
        r'P.0 B0X': 'P.O. Box',
        r'NYERI BRANCH': 'Nyeri Branch',
        r'PIN N0': 'PIN No',
        r'HRA/EFP': 'KRA/EFP',
        r'IILL NO.': 'BILL NO.',
        r'CASH SALE#': 'Cash Sale #',
        r'DATE :': 'Date :',
        r'TIME :': 'Time :',
        r'BRANCH:01': 'Branch:01',
        r'ITEM': 'Item',
        r'QTY': 'Qty',
        r'PRICE': 'Price',
        r'AMOUNT': 'Amount',
        r'MAYCORN MAIZE MEAL2KG': 'Maycorn Maize Meal 2KG',
        r'CASH': 'Cash',
        r'CHANGE': 'Change',
        r'ABC Company': 'ABC Company',
        r'POS Invoice': 'POS Invoice',
        r'Bill No.': 'Bill No.',
        r'Time': 'Time',
        r'Date': 'Date',
        r'SI Description': 'SI Description',
        r'Qty': 'Qty',
        r'Rate': 'Rate',
        r'Amount': 'Amount',
        r'Laptops': 'Laptops',
        r'VAT': 'VAT',
        r'Total': 'Total',
        r'Cash': 'Cash',
        r'Cash Tendered': 'Cash Tendered',
        r'Balance': 'Balance',
        r'TotalPaid': 'Total Paid',
        r'ThankYou!': 'Thank You!',
        r'Visit Again!': 'Visit Again!',
        r'Mkatrungan po b ang presyo.pang aerox.': 'Mkatrungan po b ang presyo pang aerox.',
        r'motorcycleshop santiagogentri': 'Motorcycle Shop Santiago Gentri',
        r'NO.': 'No.',
        r'DATE': 'Date',
        r'SOLDTO': 'Sold To',
        r'ADDRESS': 'Address',
        r'QTY.UNIT': 'Qty. Unit',
        r'ARTICLES': 'Articles',
        r'PRICE': 'Price',
        r'AMOUNT': 'Amount',
        r'259D': '259D',
        r'PishonRyng': 'Pishon Ring',
        r'BasuloachiA': 'Basuloachi A',
        r'HadbaskeA': 'Hadbaske A',
        r'VolueSeala': 'Value Scale',
        r'Cams': 'Cams',
        r'Pluilere': 'Pluilere',
        r'NES': 'NES',
        r'epdnA': 'epdn A',
        r'Zlioo': 'Zlioo',
        r'DL': 'DL',
        r'nleD': 'nle D',
        r'DRvO': 'DRvO',
        r'ho': 'ho',
        r'TOTAL': 'Total',
        r'nDFax': 'Fax',
        r'GENERAL MISCELLANEOUS INCOME': 'General Miscellaneous Income',
        r'Date 140112026': 'Date 140112026',
        r'No.': 'No.',
        r'Joseph': 'Joseph',
        r'Received from': 'Received from',
        r'For': 'For',
        r'Acoesr': 'Acoesr',
        r'Tweaty only': 'Twenty only',
        r'The Sum ofShs': 'The Sum of Shs',
        r'T·LB': 'T. LB',
        r'onaccount': 'on account',
        r'Kshis': 'KSh',
        r'Sign:': 'Sign:',
        r'Cash/Cheque No': 'Cash/Cheque No',
        r'For:County Treasury': 'For: County Treasury',
        r'C0151331667': 'C0151331667',
        r'NANYUKI BRANCH': 'Nanyuki Branch',
        r'jen Shoppinglomplex': 'Jen Shopping Complex',
        r'Roysambu': 'Roysambu',
        r'Next toPamkibuse': 'Next to Pamkibuse',
        r'Gefro Imani Building': 'Gefro Imani Building',
        r'Royal Plaza Building': 'Royal Plaza Building',
        r'NexttoKANU Grounds': 'Next to KANU Grounds',
        r'PO.Box121410109Nyeni': 'P.O. Box 1214-10109 Nyeri',
        r'5THFloorRm No.510': '5th Floor Rm No.510',
        r'1ST Floor Rm No.1': '1st Floor Rm No.1',
        r'Tel:0718545361': 'Tel:0718545361',
        r'Tel:0724243338/0739-347199': 'Tel:0724243338/0739-347199',
        r'0735166752': '0735166752',
        r'No!': 'No!',
        r'827119': '827119',
        r'Date:': 'Date:',
        r'20.2.6': '20.2.6',
        r'The Sum of Shillings': 'The Sum of Shillings',
        r'厦': '',
        r'V': '',
        r'a.': '',
        r'Being payment': 'Being payment',
        r'of': 'of',
        r'2412': '2412',
        r'203': '203',
        r'Kshs.': 'KSh',
        r'With Thanks': 'With Thanks',
        r'CashyCheque': 'Cash/Cheque',
        r'Landlord/ Lady': 'Landlord/Lady',
        r'Signature': 'Signature',
        r'LeGu': 'LeGu',
        r'h': 'h',
        r'Rm. No.': 'Rm. No.',
        r'?Naivas Umoja': 'Naivas Umoja',
        r'UAT:0109300U': 'UAT:0109300U',
        r'PIN:P051123223G': 'PIN: P051123223G',
        r'30606USER:': '30606 USER:',
        r'02011': '02011',
        r'DateTime:11/27/2015 4:19:03 PM St0re:27': 'DateTime: 11/27/2015 4:19:03 PM Store:27',
        r'PRICE': 'Price',
        r'AMOUNT': 'Amount',
        r'ITEM': 'Item',
        r'QTY': 'Qty',
        r'1x 199.00': '1x 199.00',
        r'15000615': '15000615',
        r'199.00A': '199.00 A',
        r'MAUS BABY OIL 238ML': 'Maus Baby Oil 238ML',
        r'X': 'X',
        r'85.00': '85.00',
        r'13505790': '13505790',
        r'85.08B': '85.08 B',
        r'NAIUAS WHOLEMEAL BREAD 8OOGMS': 'Naivas Wholemeal Bread 800gms',
        r'1x219.00': '1x219.00',
        r'16500851': '16500851',
        r'219.00 A': '219.00 A',
        r'EXCELORANGEJUICE2LTR': 'Excel Orange Juice 2Ltr',
        r'OTAL': 'Total',
        r'503.00': '503.00',
        r'HSE': 'HSE',
        r'1,000.00': '1,000.00',
        r'HANGE': 'Change',
        r'497.00': '497.00',
        r'TAL': 'Total',
        r'ITEMS: 3': 'Items: 3',
        r'ODE': 'ODE',
        r'RATE': 'Rate',
        r'UATABLE AMT': 'UATable Amt',
        r'VAT AMT': 'VAT Amt',
        r'16.00%': '16.00%',
        r'360.34': '360.34',
        r'57.66': '57.66',
        r'0.00%': '0.00%',
        r'85.00': '85.00',
        r'0.00': '0.00',
        r'ash': 'Cash',
        r'1000': '1000',
        r'Thank': 'Thank',
        r'KYou!': 'You!',
        r'27-11-201516:20:22': '27-11-2015 16:20:22',
        r'RECEIPT#00113199': 'Receipt#00113199',
        r'FISCAL RECEIPT': 'Fiscal Receipt',
        r'M/C ID H AD30103678OF IKHETIA\'S BUSIA': 'M/C ID H AD30103678 OF IKETIA\'S BUSIA',
        r'P.0.Box 668.Kitale,Kenya': 'P.O. Box 668 Kitale, Kenya',
        r'CASH SALE': 'Cash Sale',
        r'VAT Reg:00014695QP1n No.P000628476L': 'VAT Reg:00014695Q Pin No.P000628476L',
        r'Date:03:52pm Thu 01 Apr112021': 'Date:03:52pm Thu 01 April 2021',
        r'Branch:34T:5 Session:980 Rct:1.6': 'Branch:34 T:5 Session:980 Rct:1.6',
        r'2VDA4FRS': '2VDA4FRS',
        r'2104010509800126': '2104010509800126',
        r'Item Qty': 'Item Qty',
        r'Each': 'Each',
        r'Total': 'Total',
        r'MAZIWA MALA KCC 5OOML BTL RND': 'Maziwa Mala KCC 500ML BTL RND',
        r'G': 'G',
        r'416840': '416840',
        r'1.000 PCS': '1.000 PCS',
        r'72.00': '72.00',
        r'72.00': '72.00',
        r'BODY-LOTION VENUS NOURIS.24HR 40OML': 'Body Lotion Venus Nouris.24HR 400ML',
        r'G': 'G',
        r'210937': '210937',
        r'1.000 PCS': '1.000 PCS',
        r'249.00': '249.00',
        r'249.00': '249.00',
       
        r'G': 'G',
        r'150168': '150168',
        r'1.000PKI': '1.000 PKI',
        r'96.00': '96.00',
        r'96.00': '96.00',
        r'BISCUIT | SUPA VANILLA SMALL 290G': 'Biscuit Supa Vanilla Small 290G',
        r'G': 'G',
        r'495517': '495517',
        r'1.000PKT': '1.000 PKT',
        r'57.00': '57.00',
        r'57.00': '57.00',
        r'BAGS VEST | NON-W FABRIC S-2235GS': 'Bags Vest Non-W Fabric S-2235GS',
        r'SPLN PIG': 'Spln Pig',
        r'270322': '270322',
        r'1.000 PCS': '1.000 PCS',
        r'15.00': '15.00',
        r'15.00': '15.00',
        r'TOTAL': 'Total',
        r'：': '',
        r'489.00': '489.00',
        r'CASH PAID': 'Cash Paid',
        r'=': '',
        r'1.000.00': '1,000.00',
        r'CHANGE': 'Change',
        r'|': '',
        r'511.00': '511.00',
        r'TotalQty:5.00units': 'Total Qty: 5.00 units',
        r'PXU OCP': 'PXU OCP',
        r'Gross Wt:1.11 kgs （approx.)': 'Gross Wt: 1.11 kgs (approx.)',
        r'Cashier:ALLOYS': 'Cashier: Alloys',
        r'Supervisor: StanleyKE': 'Supervisor: Stanley KE',
        r'###START OF LEGALLEIPT ##': '### START OF LEGAL RECEIPT ##',
        r'REGISTER N0:KE8000314380849': 'Register No: KE8000314380849',
        r'TUMAINI SELF SERUICF-': 'Tumaini Self Service',
        r'P.0.B0X339-050': 'P.O. Box 339-050',
        r'RVICELTD': 'Service Ltd',
        r'TEL:029-279851': 'Tel:029-279851',
        r'g0.80X339-00507': 'g0.80X339-00507',
        r'团': '',
        r'0327#A': '0327#A',
        r'00430459': '00430459',
        r'USR 00000001': 'USR 00000001',
        r'KRA/EFP/01052013/20257B': 'KRA/EFP/01052013/20257B',
        r'DATE:01/03/2018': 'Date:01/03/2018',
        r'RCT#：0515000327': 'Rct#:0515000327',
        r'CODE': 'Code',
        r'DESCRIPTION': 'Description',
        r'QTY': 'Qty',
        r'OPRICE AMOUNT VAT': 'Price Amount VAT',
        r'0EE': '0EE',
        r'0100055': '0100055',
        r'2X': '2X',
        r'520.88': '520.88',
        r'FRESH FRI': 'Fresh Fri',
        r'C/0IL3L': 'C/Oil 3L',
        r'1,040.88A': '1,040.88 A',
        r'0200096': '0200096',
        r'TA': 'TA',
        r'30.00': '30.00',
        r'KENSALT TABLE SALT 1KG': 'Kensalt Table Salt 1KG',
        r'30.00A': '30.00 A',
        r'0200129': '0200129',
        r'1X': '1X',
        r'85.00': '85.00',
        r'PEPTANG TOMATO SAUCE 400G': 'Peptang Tomato Sauce 400G',
        r'85.00A': '85.00 A',
        r'0200653': '0200653',
        r'1X': '1X',
        r'85.00': '85.00',
        r'ROYCO MCHUZI MIX 2OOG SAICHET': 'Royco Mchuzi Mix 200g Saichet',
        r'85.00A': '85.00 A',
        r'0308295': '0308295',
        r'1X': '1X',
        r'100.00': '100.00',
        r'JIK LEMON 250ML': 'Jik Lemon 250ML',
        r'100.00A': '100.00 A',
        r'0300408': '0300408',
        r'2X': '2X',
        r'70.00': '70.00',
        r'DETTOL SOAP COOL 60G': 'Dettol Soap Cool 60G',
        r'140.00A': '140.00 A',
        r'0301551': '0301551',
        r'XT': 'XT',
        r'290.00': '290.00',
        r'ARIEL DOWNY W/POWDER 1KG': 'Ariel Downy w/Powder 1KG',
        r'290.00A': '290.00 A',
        r'1X': '1X',
        r'225.00': '225.00',
        r'0400031': '0400031',
        r'225.00A': '225.00 A',
        r'KERICHO GOLD PREM.T/BAGS 100'S': 'Kericho Gold Prem. T/Bags 100\'s',
        r'0400213': '0400213',
        r'2X': '2X',
        r'55.00': '55.00',
        r'RAHA DRINKING CHOCOLATE 100G': 'Raha Drinking Chocolate 100G',
        r'110.00A': '110.00 A',
        r'120.00': '120.00',
        r'0500025': '0500025',
        r'120.00A': '120.00 A',
        r'AQUAFRESH BIG TEETH 5OML': 'Aquafresh Big Teeth 50ML',
        r'200.00': '200.00',
        r'0500108': '0500108',
        r'200.': '200.',
        r'COLG.T/PASTE MCP 100ML/14': 'Colg. T/Paste MCP 100ML/14',
        r'545': '545',
        r'0600819': '0600819',
        r'.5DAY CREAN': '.5 Day Cream',
        r'545': '545',
        r'JOHNSONS': 'Johnsons',
        r'23': '23',
        r'0601066': '0601066',
        r'VASELU': 'Vaseline',
        r'D/SKIN NE': 'D/Skin NE',
        r'2': '2',
        r'0601': '0601',
        r'ARI': 'Ari',
        r'NGJELLY': 'NG Jelly',
        r'06naivas': '06 Naivas',
        r'|': '',
        r'14201:65': '14201:65',
        r'Z': 'Z',
        r'LEL:NATUASHA': 'Lel: Natuasha',
        r'·': '',
        r'93000.00A': '93000.00 A',
        r'AeIE.B0...': 'AeIE B0...',
        r'|': '',
        r'93800.00': '93800.00',
        r'1021 0812': '1021 0812',
        r'FISCALRECETPT': 'Fiscal Receipt',
        r'15:59': '15:59',
        r'NS': 'NS',
        r'20844406': '20844406',
        r'P.O.BDX10': 'P.O. Box 10',
        r'-': '',
        r'|': '',
        r'.0.0': '.0.0',
        r'STRIPT': 'Stript',
        r'08 8': '08 8',
        r'OET': 'OET',
        r'09-05-19': '09-05-19',
        r'THANKYOUI11': 'Thank You!',
        r'14:1B': '14:1B',
        r'SN': 'SN',
        r'NON-FTSOAL RECE!PTVAT:0145571K': 'Non-Fiscal Receipt VAT:0145571K',
        r'KSHS': 'KSh',
        r'DEPT 01 E0%': 'Dept 01 E0%',
        r'6.100.0': '6,100.0',
        r'SUB-TOTALEO%': 'Sub-Total E0%',
        r'6.100.0': '6,100.0',
        r'NET TOTAL': 'Net Total',
        r'6,100.0': '6,100.0',
        r'SUM OFVAT': 'Sum of VAT',
        r'0.0': '0.0',
        r'TOTAL KSHS': 'Total KSh',
        r'6.100.0': '6,100.0',
        r'CASH': 'Cash',
        r'6.100.0': '6,100.0',
        r'CLK#1': 'Clk#1',
        r'PC#1': 'PC#1',
        r'N.0001': 'N.0001',
        r'04/05/19': '04/05/19',
        r'14:56': '14:56',
        r'GD 72800010': 'GD 72800010',
    }
    
    
    
    # Issuer-specific
    if issuer == 'Munch Lavington':
        replacements.update({
            r'BURUBURU BRANCH': 'Buruburu Branch',
            r'Vi1lage Market': 'Village Market',
            r'MUNCH LAVINGTON': 'Munch Lavington',
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
# Extract key fields
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
    
    data['receipt_number'] = re.search(r'\b\d{6,}\b', all_text).group(0) if re.search(r'\b\d{6,}\b', all_text) else None
    data['date'] = re.search(r'\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\'?\d{2,4}', all_text, re.I).group(0) if re.search(r'\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\'?\d{2,4}', all_text, re.I) else None
    data['total_ksh'] = re.search(r'(subtotal|total|due)\s*kes?\s*(\d+[,.\d]*)', all_text, re.I).group(2) if re.search(r'(subtotal|total|due)\s*kes?\s*(\d+[,.\d]*)', all_text, re.I) else None
    
    if issuer == 'Munch Lavington':
        data['branch'] = re.search(r'(\w+ branch)', all_text, re.I).group(1) if re.search(r'(\w+ branch)', all_text, re.I) else None
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
