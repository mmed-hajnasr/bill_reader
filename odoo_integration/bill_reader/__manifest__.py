# -*- coding: utf-8 -*-
# Part of Odoo. See LICENSE file for full copyright and licensing details.

{
    'name': 'bill reader',
    'version': '1.0',
    'category': 'Uncategorized',
    'summary': 'Python program to extract information from receipt images',
    'description': """
bill reader
============================

Bill reader is a Python program designed to extract relevant information from images of receipts. It utilizes computer vision techniques and Optical Character Recognition (OCR) to identify and extract key details such as the vendor's name, date of purchase, total amount, and individual items.

This module includes:
- Standalone Receipt Parser functionality.
- Integration with Odoo for automated receipt data extraction.

Credits
-------
Author: Mohamed Ben Hadj Nasr

Acknowledgments
---------------
We would like to thank SRA integration and mohamed najjari for their support and guidance during the development of the Odoo integration.

""",
    'author': 'Mohamed Ben Hadj Nasr',
    'depends': [
        'base','hr_expense'
    ],
    'data': [
        'views/main_menu.xml'
    ],
    'installable': True,
    'application': True,
    'auto_install': False,
}