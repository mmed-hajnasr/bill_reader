from pdf2image import convert_from_path
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pytesseract import Output
import pytesseract as ts


def get_image(file):
    if file.endswith(".pdf"):
        full_image = convert_from_path(file)
        image = full_image[0]
    else:
        image = Image.open(file)
    return image


def isprice(text) -> bool:
    curr = ""
    point = False
    for i, c in enumerate(text):
        if (c == "$" or c == "€") and i == 0:
            curr = c
        if c.isdigit():
            continue
        if c == ".":
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
