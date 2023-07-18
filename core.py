from pdf2image import convert_from_path
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pytesseract import Output
import pytesseract as ts
import cv2


def get_image(file):
    if file.endswith(".pdf"):
        full_image = convert_from_path(file)
        image = full_image[0]
    else:
        image = Image.open(file)
    image = np.array(image)
    return image


def isprice(text) -> bool:
    curr = ""
    point = False
    for i, c in enumerate(text):
        if (c == "$" or c == "€") and i == 0:
            curr = c
            continue
        if c.isdigit():
            continue
        if c == "." or c ==",":
            if (
                point == False
                and i != 0
                and i != len(text) + 1
                and text[i - 1].isdigit()
                and text[i + 1].isdigit()
            ):
                point = True
            else:
                return False
        elif curr == "":
            curr = text[i:]
            break
        else:
            return False
    if len(curr) > 4:
        return False
    else:
        return point

def is_name(word):
    return not isprice(word) and not word.isdigit()

def merge_names(line):
    ind = 0
    while ind < (len(line)-1) :
        while ind+1 < len(line) and is_name(line[ind]) and is_name(line[ind+1]):
            line[ind] += ' ' + line[ind+1]
            line.pop(ind+1)
        ind += 1

def same_line(A,B):
    if abs(A-B) < 10 :
        return True 
    else:
        return False

def clean_the_text(row):
    row.text = "".join([c if ord(c) < 128 else "" for c in row.text]).strip()
    row.text.replace("\n", "")
    return row