# bill_reader - Receipt Parser for Odoo Integration

## introduction 
Welcome to the Receipt Parser! This Python program is designed to extract relevant information from images of receipts. It utilizes computer vision techniques and Optical Character Recognition (OCR) to identify and extract key details such as the vendor's name, date of purchase, total amount, and individual items.

### Odoo Integration Module:
In addition to its standalone functionality, bill_reader offers seamless integration with Odoo, a popular open-source ERP system. With the Odoo integration module, users can directly utilize the Receipt Parser within their Odoo environment. This allows for automated receipt data extraction and integration into various financial and accounting processes in Odoo.

### Supported Image Formats:
The program supports various image formats, including pdf, png, jpg, and webp, ensuring flexibility and compatibility with different types of receipt images.

With the power of computer vision and OCR, bill_reader simplifies the process of capturing and organizing financial data from paper receipts, streamlining expense tracking and accounting tasks.

## Getting Started

Before you begin using the Receipt Parser (bill_reader) and its Odoo integration module, make sure you have the following prerequisites in place:

### Tesseract Installation:
Tesseract is an OCR engine required for text recognition in images. Ensure you have Tesseract installed on your computer before using the Receipt Parser. You can download Tesseract from the official website (https://github.com/tesseract-ocr/tesseract) or install it using package managers like apt (for Linux) or Homebrew (for macOS).

### Standalone Version:
If you wish to use the Receipt Parser as a standalone Python program, navigate to the project's root directory and install the necessary dependencies using the provided requirements.txt file:

```
cd bill_reader
pip install -r requirements.txt
```

This will install all the required Python packages to run the standalone version.

### Odoo Integration:
If you want to integrate the Receipt Parser with Odoo, you will need to follow these steps:

First, ensure you have set up Odoo and have the necessary access rights to install modules.

Next, navigate to the bill_reader/odoo_integration directory and install the required dependencies using the provided requirements.txt file:

```
cd bill_reader/odoo_integration
pip install -r requirements.txt
```

**Note:** The second requirements.txt should be installed within the Odoo environment.

With the prerequisites and dependencies in place, you are now ready to use the Receipt Parser as a standalone program or integrate it with your Odoo environment.

## Files and their Purposes

- data folder:

**Description**: This folder contains sample data that you can use to test the Receipt Parser. It includes multiple receipts from different places, providing a diverse set of test cases to evaluate the program's performance.

- core.py:

**Description**: The core.py file is the heart of the Receipt Parser. It houses all the essential functions and algorithms necessary for extracting relevant information from the receipt images. These functions utilize computer vision techniques and Optical Character Recognition (OCR) to identify and extract key details such as the vendor's name, date of purchase, total amount, and individual items.

- demo.ipynb:

**Description**: The demo.ipynb file is a Jupyter Notebook that showcases how the Receipt Parser works. It provides step-by-step examples and demonstrations of how to use the core functions to process receipt images and extract information. This notebook is an excellent resource for users to get started with the program and understand its functionalities.

- is_scanned.ipynb:

**Description**: The is_scanned.ipynb file explains how the program checks whether a receipt has already been scanned or needs to be processed. It employs segmentation and other computer vision techniques to determine the presence of information on the receipt image and its processing status.

- segmentation.py:

**Description**: The segmentation.py file includes functions related to image segmentation. Image segmentation is a crucial step in the receipt processing pipeline. This file contains algorithms that divide the receipt image into meaningful regions to facilitate further analysis and data extraction.

## Acknowledgments:

I would like to extend our heartfelt gratitude to the following individuals and organizations for their support and contributions to the Receipt Parser (bill_reader) project:

SRA integration: We are thankful to SRA integration for providing an excellent learning environment and the opportunity to work on this project during our internship. Their support and resources were instrumental in the development of this program.

mohamed najjari: Special thanks to mohamed najjari for their invaluable guidance and mentorship throughout the Odoo integration process. Their expertise and encouragement played a significant role in the successful integration of the Receipt Parser with Odoo environment.

[jenswalter](https://www.kaggle.com/datasets/jenswalter/receipts): We acknowledge jenswalter for providing the diverse and extensive collection of receipt images on kaggle used as sample data in the Receipt Parser. This dataset was crucial in testing and improving the program's accuracy and performance.

The Receipt Parser would not have been possible without the support and resources provided by these individuals and organizations. I am truly grateful for their contributions.

## Conclusion

Thank you for using the Receipt Parser (bill_reader). We hope this guide has been helpful in utilizing the program's computer vision and OCR capabilities to extract valuable information from receipt images. Happy parsing!