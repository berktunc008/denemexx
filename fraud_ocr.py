import torch
import torchvision
from pytesseract import pytesseract as tess
import cv2
import fitz
import numpy as np
from PIL import Image

model_document = torch.hub.load("ultralytics/yolov5", "custom", path="document.pt") 

def tesseract_ocr(img,custom_config =r'-l eng+tur --oem 3 --psm 6' ):
    img = cv2.resize(img, None, fx=1.3, fy=1.2, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    text = tess.image_to_string(img,config=custom_config)
    return text

def text_cleaner(text,label):
    text=text.replace('\n','').replace('\x0c','').replace('|','').strip()
    if label in ['adres']:
        return text
    try:      
        list_text = text.split(' ')
    except:      
        return text 
    if len(list_text)>1:
        if label in ['kimlik_no','kimlik_no2']:
            text_ = [i for i in list_text if i.isnumeric()]
            if len(text_)==1:
                return text_[0]    
            elif len(text_)>1:
                len_text_ = [abs(len(i)-11) for i in text_]
                text = len_text_[np.argmin(len_text_)]
                return text   
            else:    
                return text
        if label in ['adres_no']:
            text_ = [i for i in list_text if i.isnumeric()]
            if len(text_)==1:
                return text_  
            elif len(text_)>1:
                len_text_ = [abs(len(i)-10) for i in text_]          
                return len_text_[np.argmin(len_text_)]         
            else:         
                return text
        if label in ['belge_no']:
            try:
                first_loc = text.find('-')
                return text[first_loc-4:first_loc+15]            
            except:
                return text    
    else :
        return text
    
def document_f(image_path, model=model_document):
    results = model_document(image_path)
    doc_results = results.crop(save=False)
    doc_ocr = dict()
    for i,j in enumerate(doc_results):
        if float(doc_results[i]['label'].split(' ')[1])>0.5:
            label = doc_results[i]['label'].split(' ')[0]
            cropped_im = doc_results[i]['im']
            cropped_ocr = tesseract_ocr(cropped_im)
            doc_ocr[label] = str(text_cleaner(cropped_ocr,label))
    return doc_ocr

def fraud_ocr(img):
    ocr_result = document_f(img)
    if ocr_result.get('kimlik_no'):
        ocr_tckn = ocr_result['kimlik_no']
    elif ocr_result.get('kimlik_no2'):
        ocr_tckn = ocr_result['kimlik_no2']   
    else:
        ocr_tckn = None
    return {"tc": ocr_tckn, "name": ocr_result.get('ad'), "surname": ocr_result.get('soyad')}