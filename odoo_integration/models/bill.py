from odoo import models,fields
from tkinter import filedialog


class bill(models.Model):
    _inherit = 'hr.expense'

    upload_file = fields.Binary(string='Upload file')
    
    def does_nothing(self):
        file_path = self.env['ir.attachment'].with_context(binary_field='file_data').open_file_dialog()
        print(file_path)
