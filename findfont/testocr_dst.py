# -*- encoding: utf-8 -*-
from PIL import Image
from pytesseract import *
import numpy as np
import cv2 as cv
import findfont.image_handler as ih


def detect_text(filename, save=False):
    # def detect_text(imageArr, save=False):
    # filename = r'../images/CaptureFont/beer_mo1.png'
    image = Image.open(filename)
    # print(image.shape)
    image = np.array(image)
    print(image.shape)
    # dst = cv.resize(image, dsize=(800,800), interpolation=cv.INTER_CUBIC)
    # print(type(dst), dst.shape)
    # grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ret, img_result = cv.threshold(grayscale, 200, 255, cv.THRESH_BINARY)
    # blur = cv.blur(image, (3,3), anchor=(-1,-1), borderType=cv.BORDER_REFLECT_101)
    # cv.imshow("img", dst)
    # cv.waitKey()
    # cv.imshow("img", image)
    # cv.waitKey()
    text = image_to_string(image, lang='kor', config='--psm 10')
    # text = image_to_string(imageArr, lang='kor+eng', config='--psm 10')
    # print("[", text, "]")
    if save:
        with open('D:/pyproject/bitopencv/images/poster_ocr/mo_ocr.txt', 'w') as f:
            f.write(text)
    return text

def detect_text2(imageArray, save=False):
    image = Image.fromarray(imageArray)
    # print(image.shape)
    image = np.array(image)
    print(image.shape)
    # filename = r'../images/CaptureFont/beer_mo1.png'
    # image = np.array(imageArray)
    # dst = cv.resize(image, dsize=(800,800), interpolation=cv.INTER_CUBIC)
    # print(type(dst), dst.shape)
    # grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ret, img_result = cv.threshold(grayscale, 200, 255, cv.THRESH_BINARY)
    # blur = cv.blur(img_result, (3,3), anchor=(-1,-1), borderType=cv.BORDER_REFLECT_101)
    # cv.imshow("img", dst)
    # cv.waitKey()
    # cv.imshow("img", image)
    # cv.waitKey()

    # ih.show_image(imageArray)

    text = image_to_string(image, lang='kor', config='--psm 10')
    print("[", text, "]")
    if save:
        with open('D:/pyproject/bitopencv/images/poster_ocr/mo_ocr.txt', 'w') as f:
            f.write(text)
    return text


# if __name__ == "__main__":
#    text = detect_text(r'../images/CaptureFont/pre_jo.png')