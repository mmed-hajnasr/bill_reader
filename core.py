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
        if c == "." or c == ",":
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
    while ind < (len(line) - 1):
        while ind + 1 < len(line) and is_name(line[ind]) and is_name(line[ind + 1]):
            line[ind] += " " + line[ind + 1]
            line.pop(ind + 1)
        ind += 1


def same_line(A, B):
    if abs(A - B) < 10:
        return True
    else:
        return False


def clean_the_text(row):
    row.text = "".join([c if ord(c) < 128 else "" for c in row.text]).strip()
    row.text.replace("\n", "")
    return row


def get_match(element):
    if element.isdigit():
        return "quantity"
    elif isprice(element):
        return "price"
    else:
        return "desc"


def get_format(line):
    line_format = []
    for element in line:
        if element.isdigit():
            if line_format.count("quantity") >= 1:
                continue
            line_format.append("quantity")
        elif isprice(element):
            if line_format.count("price") >= 2:
                continue
            line_format.append("price")
        else:
            if line_format.count("desc") >= 1:
                continue
            line_format.append("desc")
    return line_format


def follow_format(line, line_format):
    line_index = 0
    format_index = 0
    while line_index < len(line) and format_index < len(line_format):
        if get_match(line[line_index]) == line_format[format_index]:
            line_index += 1
            format_index += 1
        else:
            line_index += 1
    return format_index == len(line_format)


def get_data(line, line_format):
    line_index = 0
    format_index = 0
    row = []
    while line_index < len(line) and format_index < len(line_format):
        if get_match(line[line_index]) == line_format[format_index]:
            row.append(line[line_index])
            line_index += 1
            format_index += 1
        else:
            line_index += 1
    return row

def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
    return num

def isolate_products(product_lines):
    lines =[]
    last_line = -1
    distances = []
    for line in product_lines:
        lines.append(line)
        if last_line == -1:
            last_line = line
            continue
        distances.append(line - last_line)
        last_line = line
    step = most_frequent(distances)
    for ind in range(len(lines)):
        neighbours_count = 0
        if ind != 0 :
            if lines[ind] - lines[ind-1] < step + 10 :
                neighbours_count +=1
        if ind!=len(lines)-1 :
            if lines[ind+1] - lines[ind] < step + 10 :
                neighbours_count +=1
        if neighbours_count==0:
            product_lines.pop(lines[ind])
    
def get_lines(df):
    product_lines = {}
    prices = df[df['text'].apply(isprice)]
    for x in prices.top.values:
        product_lines[x] = []
    for ind,row in df.iterrows():
        for line in product_lines:
            if same_line(row.top,line) :
                product_lines[line].append(row.text)
        for line in product_lines:
            merge_names(product_lines[line])   
    return product_lines