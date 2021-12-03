import numpy as np
import os
from tqdm import tqdm
import random
import json

filename = 'dataset.json'
directory = '/workspace/melanoma_isic_dataset/stylegan2-ada-pytorch/processed_dataset_256'

with open(os.path.join(directory, filename)) as file:
    data = json.load(file)['labels']

    for img in tqdm(data):
        img_dir = os.path.join(directory,img[0])
        label = img[1]

        execute = "python projector.py "
        execute = execute + " --outdir=./projector"
        execute = execute + " --target=" + img_dir
        execute = execute + " --network=/workspace/melanoma_isic_dataset/stylegan2-ada-pytorch/training_runs/network-snapshot-020000.pkl"
        execute = execute + " --class_label " + str(label)
        execute = execute + " --num-steps 1000"

        #print(execute)
        os.system(execute)
        #exit(-1)
