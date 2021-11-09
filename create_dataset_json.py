import json
from pathlib import Path
import pandas as pd

CSV_DIR = Path('/media/14TBDISK/sandra/test_isic/')
train_df = pd.read_csv(CSV_DIR/'train_concat.csv')
labels_list = []
for n in range(len(train_df)):
    labels_list.append([train_df.iloc[n].image_name,int(train_df.iloc[n].target)])

labels_list_dict = { "labels" : labels_list}

with open("labels.json", "w") as outfile:
    json.dump(labels_list_dict, outfile)