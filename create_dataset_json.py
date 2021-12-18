import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os 

""" 
CSV_DIR = Path('/home/Data/melanoma_external_256/')
df = pd.read_csv(CSV_DIR/'train_concat.csv')
train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42)
train_df=pd.DataFrame(train_split)

labels_list = []
for n in range(len(train_df)):
    labels_list.append([train_df.iloc[n].image_name,int(train_df.iloc[n].target)])
"""


labels_list = []
input_images = [str(f) for f in sorted(Path('/workspace/melanoma_isic_dataset/all_melanoma/SAM_Dataset').rglob('*')) if os.path.isfile(f)]
for img_path in input_images:
    label = img_path.split('/')[-2]
    if label == 'In_situ':
        labels_list.append([img_path, '0'])
    else:
        labels_list.append([img_path, '1'])

labels_list_dict = { "labels" : labels_list}
with open("/workspace/melanoma_isic_dataset/all_melanoma/SAM_Dataset/labels.json", "w") as outfile:
    json.dump(labels_list_dict, outfile)
