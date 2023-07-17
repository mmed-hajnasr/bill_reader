import os

import shutil

directory = '/home/mmed/Downloads/test/'
target_dir = '/home/mmed/Documents/projects/bill_reader/data'
for filename in os.listdir(directory):
    if filename.isnumeric():
        cur_dir = os.path.join(directory, filename)
        for country in os.listdir(cur_dir):
            if country == "us":
                final_dir = os.path.join(cur_dir, country)
                for institute in os.listdir(final_dir):
                    if institute == "restaurant" or institute == "retail":
                        final_final_dir = os.path.join(final_dir, institute)
                        for file in os.listdir(final_final_dir):
                            shutil.move(os.path.join(final_final_dir,file),os.path.join(target_dir, file))