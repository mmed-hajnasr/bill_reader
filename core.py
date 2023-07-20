from pdf2image import convert_from_path
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pytesseract import Output
import pytesseract as ts
import cv2
import math


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
                and i != len(text) - 1
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
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    return num


def isolate_products(product_lines):
    lines = []
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
        if ind != 0:
            if lines[ind] - lines[ind - 1] < step + 10:
                neighbours_count += 1
        if ind != len(lines) - 1:
            if lines[ind + 1] - lines[ind] < step + 10:
                neighbours_count += 1
        if neighbours_count == 0:
            product_lines.pop(lines[ind])


def get_lines(df):
    product_lines = {}
    prices = df[df["text"].apply(isprice)]
    for x in prices.top.values:
        product_lines[x] = []
    for ind, row in df.iterrows():
        for line in product_lines:
            if same_line(row.top, line):
                product_lines[line].append(row.text)
        for line in product_lines:
            merge_names(product_lines[line])
    return product_lines


def del_suffix(product_lines):
    for line in product_lines:
        for element in reversed(product_lines[line]):
            if not isprice(element):
                product_lines[line].remove(element)
            else:
                break


def contain(List, sub):
    for s in List:
        if sub in s.lower():
            return True
    return False


def strip_price(raw_price):
    price = ""
    for c in raw_price:
        if c == ",":
            price += "."
            continue
        if c.isdigit() or c == ".":
            price += c
    return float(price)


def process_additional_data(product_lines):
    additional_data = {}
    max_ind = -1
    data = []
    taxes = []
    totals = []
    calculated_total = 0
    start_collecting = False
    additional_data["state"] = ""
    for ind in product_lines:
        if contain(product_lines[ind], "tax"):
            taxes.append(strip_price(product_lines[ind][-1]))
            start_collecting = True
            data.append(ind)
            continue
        elif contain(product_lines[ind], "total") or contain(
            product_lines[ind], "balance"
        ):
            totals.append(strip_price(product_lines[ind][-1]))
            data.append(ind)
            start_collecting = True
            if len(totals) == 2:
                start_collecting = False
            continue
        elif start_collecting:
            taxes.append(strip_price(product_lines[ind][-1]))
            data.append(ind)
    if data:
        max_ind = min(data)
    else :
        max_ind = max(product_lines) + 1
    for ind in product_lines:
        if ind < max_ind:
            calculated_total += strip_price(product_lines[ind][-1])
    tax = sum(taxes)
    if len(totals) == 2:
        if not math.isclose(totals[0], calculated_total, abs_tol=0.2):
            additional_data[
                "state"
            ] += "carefull there might be an error in the products section\n"
        additional_data["subtotal"] = totals[0]
        additional_data["costs"] = tax
        additional_data["total"] = totals[0] + tax
        if not math.isclose(totals[1] - totals[0], tax, abs_tol=0.2):
            additional_data[
                "state"
            ] += "carefull there might be an error in the taxes section\n"
    elif len(totals) == 1:
        if math.isclose(totals[0], calculated_total, abs_tol=0.2):
            additional_data["subtotal"] = totals[0]
            additional_data["tax"] = tax
            additional_data["total"] = tax + totals[0]
        elif math.isclose(totals[0], calculated_total + tax, abs_tol=0.2):
            additional_data["total"] = totals[0]
            additional_data["tax"] = tax
            additional_data["subtotal"] = totals[0] - tax
        else:
            additional_data[
                "state"
            ] += "carefull there might be an error in reading the bill \n"
            additional_data["total"] = calculated_total
            additional_data["tax"] = tax
            additional_data["subtotal"] = totals[0] - tax
    else:
        additional_data[
            "state"
        ] += "carefull there might be an error in reading the bill \n"
        additional_data["total"] = calculated_total
        additional_data["tax"] = tax
        additional_data["subtotal"] = calculated_total - tax
    additional_data["limit"] = int(max_ind)
    return additional_data
