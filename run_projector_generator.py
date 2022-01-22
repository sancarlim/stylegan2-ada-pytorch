#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : server_mp.py
# Modified   : 22.01.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

import numpy as np
import os
from tqdm import tqdm
import random
import json
from argparse import ArgumentParser 

random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)

parser = ArgumentParser()
parser.add_argument("--filename", type=str, default='dataset.json')
parser.add_argument("--directory", type=str, default='/workspace/stylegan2-ada-pytorch/processed_dataset_256_SAM')
parser.add_argument('--initial_tqdm', type=int, help='Restart projection', default=0)

parser.add_argument("--task", type=str, default='project', help='Choose task project/generate',
                            choices=['generate', 'project'])

parser.add_argument('--network', help='Network pickle filename', default=None)
parser.add_argument('--trunc', type=float, help='Truncation psi', default=1)
parser.add_argument('--class_idx',type=int, help='Class label (unconditional if not specified)')
parser.add_argument('--num_imgs',type=int)
parser.add_argument('--outdir', help='Where to save the output images', type=str, default=None)
 
args = parser.parse_args()

filename = args.filename
directory = args.directory

if args.task == 'project':
    with open(os.path.join(directory, filename)) as file:
        data = json.load(file)['labels']
        
        for img, label in tqdm(data, initial=args.initial_tqdm):
            img_dir = os.path.join(directory,img)
            label = 1 # FOR PROJECTING SAM DATA img[1] 

            execute = "python projector.py "
            execute = execute + " --outdir=" + args.outdir
            execute = execute + " --target=" + img_dir
            execute = execute + " --network=/workspace/stylegan2-ada-pytorch/training_runs/network-snapshot-020000.pkl"
            execute = execute + " --class_label " + str(label)
            execute = execute + " --num-steps 1000"

            #print(execute)
            os.system(execute)
            #exit(-1)
else:
    execute = "python generate.py " 
    execute = execute + " --outdir=" + args.outdir
    execute = execute + " --trunc=" + str(args.trunc)
    execute = execute + " --network=" + args.network 
    execute = execute + " --class=" + str(args.class_idx)
    execute = execute + " --seeds=" + str(np.random.randint(0,1000000,args.num_imgs)).replace('[','').replace(']','').replace(' ',',')

    os.system(execute)
