import cv2
import fitz
import numpy as np
import os
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from PIL.JpegImagePlugin import JpegImageFile
from pyzbar.pyzbar import decode
from fraud_resnext_cls_v3 import fraud_cls_file
import time

def read_edevlet_qr(file_var):
    if fraud_cls_file(file_var):
        zbar_start_time = time.time()
        edevlet_flag = False
        detections, pages = readQR_zbar(file_var)
        for detection_data, page in zip(detections, pages):
            if "barkod:" in detection_data and "tckn:" in detection_data:
                edevlet_flag = True
        print(f"---zbar-time: {time.time() - zbar_start_time}")
        if not edevlet_flag:
            cv2_start_time = time.time()
            detections, pages = readQR_cv2(file_var)
            print(f"---cv2-time: {time.time() - cv2_start_time}")
        
        edevlet_data = []
        for detection_data, page in zip(detections, pages):
            if "barkod:" in detection_data and "tckn:" in detection_data:
                index1 = detection_data.index("barkod:") + len("barkod:")
                index2 = detection_data.find(";", index1)
                barkod = detection_data[index1:index2]

                index1 = detection_data.index("tckn:") + len("tckn:")
                index2 = detection_data.find(";",index1)
                tckn = detection_data[index1:index2]

                edevlet_data.append({"barkod": barkod, "tckn": tckn, "page_img": page})
        return {"cls": True, "data": edevlet_data}
    else:
        return {"cls": False, "data": []}

def get_images(file_var):
    images = []
    if isinstance(file_var, str): #if file_var is a path
        if not os.path.exists(file_var):
            raise Exception("No such file.")
        if file_var.endswith(".pdf"):
            doc = fitz.open(file_var)
            for page in doc:
                pix = page.get_pixmap(alpha=False,matrix=fitz.Matrix(2, 2))
                pix.set_dpi(50,50)
                nparr = np.fromstring(pix.tobytes(), np.uint8)
                images.append(cv2.imdecode(nparr, cv2.IMREAD_COLOR))
        elif file_var.endswith(".tiff") or file_var.endswith(".tif"):
            with Image.open(file_var) as im:
                try:
                    while 1:
                        images.append(cv2.cvtColor(np.array(im.convert('RGB')), cv2.COLOR_RGB2BGR))
                        im.seek(im.tell() + 1)
                except EOFError:
                    pass
        elif file_var.endswith(".jpg") or file_var.endswith(".jpeg") or file_var.endswith(".png"):
            images.append(cv2.imread(file_var, cv2.IMREAD_COLOR))
        else:
            raise Exception("Unsupported file input.")
    elif isinstance(file_var, np.ndarray): #if file_var is an opencv image
        images.append(file_var)
    elif isinstance(file_var, PngImageFile) or isinstance(file_var, JpegImageFile): #if file_var is a pillow image
        images.append(cv2.cvtColor(np.array(file_var), cv2.COLOR_RGB2BGR))
    else:
        raise Exception("Unsupported file input.")
    return images

def readQR_zbar(file_var):
    get_images_start_time = time.time()
    images = get_images(file_var)
    print(f"-----get-images-time: {time.time() - get_images_start_time}")
    detections = []
    pages = []
    for image in images:
        decoded_list = decode(image) # try colored images
        if not decoded_list: # if colored doesn't work try grayscale
            decoded_list = decode(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) 
        if not decoded_list: # if grayscale doesn't work try binarization
            im_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(im_grayscale, (5, 5), 0)
            ret, bw_im = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #binarization
            decoded_list = decode(bw_im)
        if decoded_list:
            for decoded in decoded_list:
                if decoded.type == 'QRCODE':
                    detections.append(str(decoded.data))
                    pages.append(image)
    return detections, pages

def readQR_cv2(file_var):
    get_images_start_time = time.time()
    images = get_images(file_var)
    print(f"-----get-images-time: {time.time() - get_images_start_time}")
    qrCodeDetector = cv2.QRCodeDetector()
    detections = []
    pages = []
    for image in images:
        try:
            decodedText, points, _ = qrCodeDetector.detectAndDecode(image)
        except:
            points = None
        if points is None: # if colored doesn't work try grayscale
            try:
                decodedText, points, _ = qrCodeDetector.detectAndDecode(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            except:
                points = None
        if points is None: # if grayscale doesn't work try binarization
            im_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(im_grayscale, (5, 5), 0)
            ret, bw_im = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #binarization
            try:
                decodedText, points, _ = qrCodeDetector.detectAndDecode(bw_im)
            except:
                points = None
        if points is not None:
            detections.append(decodedText)
            pages.append(image)
    return detections, pages