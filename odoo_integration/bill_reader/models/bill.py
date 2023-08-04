from odoo import models,fields
from pdf2image import convert_from_path
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pytesseract import Output
import pytesseract as ts
import cv2
import math
import re
from datetime import datetime
import base64

def get_image(file):
    if file.endswith(".pdf"):
        full_image = convert_from_path(file)
        image = full_image[0]
    else:
        image = Image.open(file)
    image = np.array(image)
    return image


def isprice(text):
    text.strip()
    pattern = re.compile("^[£€\$]?\d+[,\.]\d+$|^\d+[,\.]\d+[£€\$]?$")
    if pattern.match(text):
        return not math.isclose(strip_price(text), 0, rel_tol=1e-07, abs_tol=0.0)
    return False


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
    else:
        max_ind = max(product_lines) + 1
    for ind in product_lines:
        if ind < max_ind:
            calculated_total += strip_price(product_lines[ind][-1])
    tax = sum(taxes)
    if len(totals) >= 2:
        if not math.isclose(totals[-2], calculated_total, abs_tol=0.2):
            additional_data[
                "state"
            ] += "carefull there might be an error in the products section\n"
        additional_data["subtotal"] = totals[-2]
        additional_data["costs"] = totals[-1] - totals[-2]
        additional_data["total"] = totals[-1]
        if not math.isclose(totals[1] - totals[-2], tax, abs_tol=0.2):
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
            additional_data["total"] = calculated_total + tax
            additional_data["tax"] = tax
            additional_data["subtotal"] = calculated_total
    else:
        additional_data[
            "state"
        ] += "carefull there might be an error in reading the bill \n"
        additional_data["total"] = calculated_total
        additional_data["tax"] = tax
        additional_data["subtotal"] = calculated_total - tax
    additional_data["limit"] = int(max_ind)
    return additional_data


def is_scanned(image):
    number_of_colors = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
    return number_of_colors < 1000


def get_final_image(file):
    image = get_image(file)
    if not is_scanned(image):
        image = scan(image)
    return image


def extract_dates_from_text(text):
    date_pattern = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},?\s\d{2,4}\b|\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{2,4}\b"
    matches = re.findall(date_pattern, text)
    return matches


def find_date(df):
    for word in df.text:
        if extract_dates_from_text(word):
            return extract_dates_from_text(word)[0]


def find_name(image):
    text = ts.image_to_string(image)
    lst = text.splitlines()
    for line in lst:
        cnt = sum(x.isalpha() for x in line)
        if cnt > 5:
            return line

def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype('int').tolist()

def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return order_points(destination_corners)


def scan(img):
    # Resize image to workable size
    dim_limit = 1080
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
    # Create a copy of resized original image for later use
    orig_img = img.copy()
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
 
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
 
    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    if len(page) == 0:
        return orig_img
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    # For 4 corner points being detected.
    corners = order_points(corners)
 
    destination_corners = find_dest(corners)
 
    h, w = orig_img.shape[:2]
    # Getting the homography.
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
    final = cv2.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv2.INTER_LINEAR)
    return final

import os
# start of the integration
class bill(models.Model):
    _inherit = 'hr.expense'

    upload_file = fields.Binary(string='Upload file',default = None)
    file = fields.Char(string='Upload file')
    
    @api.onchange(upload_file){
        scan_receipt()
    }
    def scan_receipt(self):
        if self.file == False:
            return
        buffer_file = '/home/odoo/python_buffer/'+self.file
        with open(buffer_file, 'wb') as f:
            decoded = base64.decodebytes(self.upload_file)
            f.write(decoded)
        image = get_final_image(buffer_file)
        os.remove(buffer_file)
        # extracting the data from the image
        results = ts.image_to_data(image, 
        output_type=Output.DICT)
        df = pd.DataFrame(data= results)
        df = df[df.text != '']
        df = df.apply(clean_the_text,axis= 1)
        product_lines = get_lines(df)
        del_suffix(product_lines)
        additional_data = process_additional_data(product_lines)
        to_del = []
        for line in product_lines:
            if line >= additional_data['limit']:
                to_del.append(line)
        for line in to_del:
            product_lines.pop(line)
        isolate_products(product_lines)
        formats = []
        for line in product_lines:
            formats.append(get_format(product_lines[line]))
        chosen_format = most_frequent(formats)
        product_list = pd.DataFrame(columns=chosen_format)
        for line in product_lines:
            if follow_format(product_lines[line],chosen_format):
                product_list.loc[len(product_list.index)] = get_data(product_lines[line],chosen_format)
        additional_data['date'] = find_date(df)
        additional_data['name'] = find_name(image)

        # putting the data in the fields

        self.name = additional_data['name']
        date_formats = ['%m/%d/%Y','%d/%m/%Y']
        for format in date_formats:        
            try:
                date = datetime.strptime(additional_data['date'], format)
                self.date = date.strftime("%Y-%m-%d")
            except:
                pass
        self.total_amount = additional_data['total']
        self.description = str(product_list)+'\n'
        if additional_data['state'] != '':
            self.description += 'Warning:\n'+additional_data['state']