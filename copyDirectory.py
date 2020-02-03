

import shutil
import os
import pandas as pd

data_path = '/home/huiying/luna/11.25/z/0/'
data_csv = '/home/huiying/luna/11.25/z/0_5075.csv'
data_save_path = '/home/huiying/luna/11.25/z/0_s/'

data_csv = pd.read_csv(data_csv)
for indx, item in data_csv.iterrows():
    item = item['filename']
    name = item[len(data_path):]
    save_path = data_save_path + name
    shutil.copytree(item, save_path)


