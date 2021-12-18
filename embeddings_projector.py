import numpy as np 
import os
from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
# import tensorflow as tf

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('/workspace/melanoma_isic_dataset/stylegan2-ada-pytorch/projector') #('/home/stylegan2-ada-pytorch/projector') #('/workspace/melanoma_isic_dataset/stylegan2-ada-pytorch/projector')

directory = "/workspace/melanoma_isic_dataset/stylegan2-ada-pytorch/projector"  #"/home/Data/generated-trunc-sameseeds" 
emb_f = "allvectors.txt"
metadata_f = "alllabelsf.txt"
transform = transforms.ToTensor()

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

with open(os.path.join(directory, emb_f)) as f:
    embeddings = f.readlines() #[::2]
embeddings_tensor = torch.tensor( [float(i) for emb_line in embeddings for i in emb_line[:-2].split(' ') ] ).reshape(len(embeddings),-1)


with open(os.path.join(directory, metadata_f)) as f:
    metadata=f.readlines() #[::2]
metadata_f = [[name.split('.')[0].split(' ')[0], name.split('.')[0].split(' ')[1]] for name in metadata]


images_pil = torch.empty(len(metadata), 3, 100,100)
labels = []
for i, line in enumerate(metadata):
    label = line.split(' ')[0]
    img_name = line.split(' ')[1].split('txt')[0] + 'from.png'  # 0 img00000552.class.0.txt 
             #line.split(' ')[1].split('.')[0] + '_' + label + '.png'    
    img_dir = os.path.join(directory,img_name)
    img = transform(Image.open(img_dir).resize((100, 100))) 
    images_pil[i] = img
    labels.append(label)

writer.add_embedding(embeddings_tensor, 
                    metadata=metadata_f,
                    metadata_header=["label","image_name"],
                    label_img=images_pil,
                    global_step=0,
                    tag='reals_with_names')
writer.close() 