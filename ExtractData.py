import os.path
import re
import subprocess
from PIL import Image
import cv2
import pytesseract
import numpy as np
import xlsxwriter

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))

INPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "images")


def findword(textlist, wordstring):
    for wordline in textlist:
        xx = wordline.split()
        if [w for w in xx if re.search(wordstring, w)]:
            lineno = textlist.index(wordline)
            textlist = textlist[lineno+1:]
            return textlist
    return textlist


def extractText(imgpath):
    path = 'media\\docs\\'+imgpath
    img1 = cv2.imread(path)
    shapeh = 1558
    shapew = 1104
    baseheight = int(shapeh)
    wsize = int(shapew)
    imageB = cv2.resize(img1, (wsize, baseheight))
    imgPath = 'static\\proof.png'
    cv2.imwrite(imgPath, imageB)
    img = Image.open(imgPath)
    img = img.convert('RGBA')
    pix = img.load()

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pix[x, y][0] < 102 or pix[x, y][1] < 102 or pix[x, y][2] < 102:
                pix[x, y] = (0, 0, 0, 255)
            else:
                pix[x, y] = (255, 255, 255, 255)

    # img.save(r'D:\temp.jpg')

    subprocess.call("tesseract " + path + " out -4", shell=True)
    text = pytesseract.image_to_string(Image.open(path), config='--psm 6 tessedit_char_whitelist=0123456789')
    text = re.sub(r'[^\x00-\x7f]+', " ", text)
    text = ''.join(filter(lambda x: ord(x) < 128, text))

    # Initializing data variable
    name = None
    passport = None
    DOB = None
    text1 = []

    lines = text.split('\n')
    for lin in lines:
        s = lin.strip()
        s = s.rstrip()
        s = s.lstrip()
        text1.append(s)

    text1 = [i for i in text1 if i]
    # print "++++++++++++++++++++++++++++++++++++"
    # print(text1)

    # Searching for NAME
    try:
        word1 = '(OFINDIA|INDIA|REPUBLIC|PUBLIC)$'
        text0 = findword(text1, word1)
        text0 = text0[1:]
        word2 = '(Country|Code|Type|Passport)$'
        text0 = findword(text0, word2)
        nameline1 = text0[0].split('_')
        passport = nameline1[len(nameline1) - 1]
        text0 = text0[1:]
        word3 = '(Given|Nameie|Name|CL|Name(s)|RawieaiGvenNames|fearaarami/GivenName(s)SOSsS=~)$'
        text0 = findword(text0, word3)
        nameline2 = text0[0].split(':')
        name = nameline2[len(nameline2) - 1]
        word4 = '(Nationality|Sex|Wallonality|GT|Date|Birth|io)'
        text0 = findword(text0, word4)
        nameline3 = text0[0].split(' ')
        DOB = nameline3[len(nameline3) - 1]
        word5 = '(ae|i|a|birth|Birth)$'
        text0 = text0[1:]
        text0 = findword(text0, word5)
        nameline4 = text0[0].split('"')
        nameline4 = nameline4[len(nameline4) - 1].split(',')
        POB = nameline4[len(nameline4) - 1]
        word6 = '(Expy|Bate|Date|Expiry|Issue|exp|expiry)$'
        text0 = text0[1:]
        text0 = findword(text0, word6)
        nameline5 = text0[0].split(' ')
        DOI = nameline5[len(nameline5) - 2]
    except Exception as ex:
        print(ex)
        pass

    img = img1[616:616 + 33, 572:572 + 187]
    img = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
    (thresh, gray) = cv2.threshold(gray, 127, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    output1 = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')

    # Making tuples of data
    data = {}
    # data['Passport Number'] = passport.replace('-. ', '').strip()
    data['Passport Number'] = output1
    data['Name'] = name.strip()
    data['Date of Birth'] = DOB.strip()
    data['Place of Birth'] = POB.strip()
    data['Date of Issue'] = DOI.replace('9', '1').strip()
    # print(data)
    workbook = xlsxwriter.Workbook('static\\proof_doc.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'Passport Number')
    worksheet.write('B1', 'Name')
    worksheet.write('C1', 'Date of Birth')
    worksheet.write('D1', 'Place of Birth')
    worksheet.write('E1', 'Date of Issue')
    worksheet.write('A2', data['Passport Number'])
    worksheet.write('B2', data['Name'])
    worksheet.write('C2', data['Date of Birth'])
    worksheet.write('D2', data['Place of Birth'])
    worksheet.write('E2', data['Date of Issue'])

    workbook.close()
    # print(data)
    return data


def display(sample, process=None):
    # cv2.namedWindow('sample', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('sample', 720, 1300)
    cv2.imshow('sample', sample)
    try:
        # cv2.namedWindow('process', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('process', 787, 336)
        cv2.imshow('process', process)
    except Exception as e:
        pass
    cv2.waitKey()
    cv2.destroyAllWindows()


def sort_contours(cnts, method="top-to-bottom"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


def extractImage(imgpath):
    path = 'media\\docs\\' + imgpath
    cropped_dir_path = r'static\\'
    img = cv2.imread(path)
    shapeh = 1558
    shapew = 1104
    baseheight = int(shapeh)
    wsize = int(shapew)
    imageB = cv2.resize(img, (wsize, baseheight))
    imgPath = 'static\\proof.png'
    cv2.imwrite(imgPath, imageB)
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    (thresh, img_bin) = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255 - img_bin
    kernel_length = np.array(img).shape[1] // 80
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=1)
    # cv2.imwrite(r"D:\\PAN\\verticle_lines.jpg", verticle_lines_img)
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=1)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=1)
    # cv2.imwrite(r"D:\\PAN\\horizontal_lines.jpg", horizontal_lines_img)
    # display(horizontal_lines_img, verticle_lines_img)
    alpha = 0.5
    beta = 1.0 - alpha
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imwrite(r"D:\\PAN\\img_final_bin.jpg", img_final_bin)
    # display(img, img_final_bin)
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    idx = 0
    width, height = img.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        idx += 1
        if 200 <= w < width / 2 and h >= 200:
            idx += 1
            new_img = img[y:y + h, x:x + w]
            cv2.imwrite(cropped_dir_path + 'profile' + '.jpg', new_img)
            y = y + h + 40
            y1 = y + 100
            x1 = x + w + 60
            new_img = img[y:y1, x:x1]
            cv2.imwrite(cropped_dir_path + 'sign' + '.jpg', new_img)
