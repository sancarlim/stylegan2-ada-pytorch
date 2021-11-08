#!/usr/bin/env python
# coding: utf-8

# Note: The io with original images is awful. Therefore I created a dataset https://www.kaggle.com/arroqc/siic-isic-224x224-images of images preprocessed to 224x224 size and saved in png (lossless).
# If you train on original jpeg large imagesm I have put the right lines of code in comments with the added comments : # Use this when training with original images

# # Pytorch Lightning Starter - SSIM Melanoma competition
# 
# I use pytorch lightning both for this competition and in my day to day work. I hope it can serve as a useful tutorial for fellow kagglers.
# Why use Pytorch-Lightning ?
# 
# Pytorch lighntning is designed to help you easily follow a pytorch based training loop and ease modifications that you may want. Want to use a new scheduler ? Then simply modify the configure_optimizer method ! The beauty of it is that it automates all the boring stuff that clogs a pure pytorch code. All these loops, .zero_grad(), .eval(), torch.save etc. are gone and handled by the framework. You just have to focus on the ML part of it. The best things for researchers is that it comes with automated logs through tensorboard to compare your many experiments and easy switches between GPU, DataParallel, TPU mixed precision etc. Obviously kaggle is not very friendly with logs so I suggest reproducing the code of this kernel in a local environment and use tensorboard there.
# 
# You may ask why not simply use fastai. This is now a matter of preference. Fastai automates a lot of stuff with best practices like .fit_one_cycle. But on the other hand unless you have a lot of experience with it I find it rather opaque in what is happening behind the scenes. It's a framework designed to go with doing the fastai course so that you understand the options. If like me you learnt deep learning in a more academic environment in pure pytorch or pure tensorflow then you may find fastai hard to understand without listening to J. Howard courses. Similarly as soon as you want to do something a bit different it can become hard to understand how to change anything. On a personal note, I'll wait for the fastai v2 course before delving into it.
# 

# In[ ]:





# 
# ## Loading data
# 
# First let's open the csv. One thing we need to make sure when splitting data in a medical context is to split by patient ID rather than image ID. Otherwise you run the risk of having some data leakage.
# 

# In[1]:


import os
import random
import argparse
from pathlib import Path
import PIL.Image as Image
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
import torch.utils.data as tdata
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


# In[2]:


# Reproductibility
SEED = 33
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[3]:


def dict_to_args(d):

    args = argparse.Namespace()

    def dict_to_args_recursive(args, d, prefix=''):
        for k, v in d.items():
            if type(v) == dict:
                dict_to_args_recursive(args, v, prefix=k)
            elif type(v) in [tuple, list]:
                continue
            else:
                if prefix:
                    args.__setattr__(prefix + '_' + k, v)
                else:
                    args.__setattr__(k, v)

    dict_to_args_recursive(args, d)
    return args


# In[77]:


CSV_DIR = Path('./melanoma_dataset/')
train_df = pd.read_csv(CSV_DIR/'train.csv')
test_df = pd.read_csv(CSV_DIR/'test.csv')
#IMAGE_DIR = Path('/kaggle/input/siim-isic-melanoma-classification/jpeg')  # Use this when training with original images
IMAGE_DIR = Path(CSV_DIR/'images224/')
#frames=[train_df, test_df]
#joint_df = pd.concat(frames)


# In[78]:


train_df.head()


# In[81]:


len(train_df)


# In[84]:


labels_list = []
for n in range(len(train_df)):
    labels_list.append([train_df.iloc[n].image_name,int(train_df.iloc[n].target)])


# In[86]:


labels_list_dict = { "labels" : labels_list}

import json
with open("labels.json", "w") as outfile:
    json.dump(labels_list_dict, outfile)


# In[ ]:


train_df.groupby(by=['patient_id'])['image_name'].count()


# In[ ]:


train_df.groupby(by=['patient_id'])['target'].mean()


# So you have patients that have multiple images. Also apparently the data is imbalanced. Let's verify:

# In[ ]:


train_df.groupby(['target']).count()


# so we have approx 60 times more negatives than positives. We need to make sure we split good/bad patients equally.

# In[ ]:


patient_means = train_df.groupby(['patient_id'])['target'].mean()
patient_ids = train_df['patient_id'].unique()


# In[ ]:


# Now let's make our split
train_idx, val_idx = train_test_split(np.arange(len(patient_ids)), stratify=(patient_means > 0), test_size=0.2)  # KFold + averaging should be much better considering how small the dataset is for malignant cases
pid_train = patient_ids[train_idx]
pid_val = patient_ids[val_idx]


# Let's verify the split was correct

# In[ ]:


train_df[train_df['patient_id'].isin(pid_train)].groupby(['target']).count()


# In[ ]:


train_df[train_df['patient_id'].isin(pid_val)].groupby(['target']).count()


# ## Pytorch Dataset
#  A dataset should simply return all the information necessary for a sample by defining the getitem and len magic methods.

# In[ ]:


class SIIMDataset(tdata.Dataset):
    
    def __init__(self, df, transform, test=False):
        self.df = df
        self.transform = transform
        self.test = test
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        meta = self.df.iloc[idx]
        #image_fn = meta['image_name'] + '.jpg'  # Use this when training with original images
        image_fn = meta['image_name'] + '.png'
        if self.test:
            img = Image.open(str(IMAGE_DIR / ('test/' + image_fn)))
        else:
            img = Image.open(str(IMAGE_DIR / ('train/' + image_fn)))
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.test:
            return {'image': img}
        else:
            return {'image': img, 'target': meta['target']}


# ## Model

# In[ ]:


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        avg_x = self.avg(x)
        max_x = self.max(x)
        return torch.cat([avg_x, max_x], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    

class Model(nn.Module):
    
    def __init__(self, c_out=1, arch='resnet34'):
        super().__init__()
        if arch == 'resnet34':
            remove_range = 2
            m = models.resnet34(pretrained=True)
        elif arch == 'seresnext50':
            m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
            remove_range = 2
            
        c_feature = list(m.children())[-1].in_features
        self.base = nn.Sequential(*list(m.children())[:-remove_range])
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(c_feature * 2, c_out))

        
    def forward(self, x):
        h = self.base(x)
        logits = self.head(h).squeeze(1)
        return logits


# 
# ## Pytorch Lightning module definition
# 
# In a normal pytorch code you probably would instantiate the model, dataloaders and make a nested for loop for epochs and batches. Pytorch lightning automates the engineering parts like the loops so that you focus on the ML part. To do that you create a pytorch lightning model and then define every ML step inside of it. To help you understand I have added comments under every method you need to implement.
# 

# In[ ]:


import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# In[ ]:


class LightModel(pl.LightningModule):

    def __init__(self, df_train, df_test, pid_train, pid_val, hparams):
        # This is where paths and options should be stored. I also store the
        # train_idx, val_idx for cross validation since the dataset are defined 
        # in the module !
        super().__init__()
        self.pid_train = pid_train
        self.pid_val = pid_val
        self.df_train = df_train

        self.model = Model(arch=hparams.arch)  # You will obviously want to make the model better :)

        self.hparams = hparams
        
        # Defining datasets here instead of in prepare_data usually solves a lot of problems for me...
        self.transform_train = transforms.Compose([#transforms.Resize((224, 224)),   # Use this when training with original images
                                              transforms.RandomHorizontalFlip(0.5),
                                              transforms.RandomVerticalFlip(0.5),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        self.transform_test = transforms.Compose([#transforms.Resize((224, 224)),   # Use this when training with original images
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.trainset = SIIMDataset(self.df_train[self.df_train['patient_id'].isin(pid_train)], self.transform_train)
        self.valset = SIIMDataset(self.df_train[self.df_train['patient_id'].isin(pid_val)], self.transform_test)
        self.testset = SIIMDataset(df_test, self.transform_test, test=True)

    def forward(self, batch):
        # What to do with a batch in a forward. Usually simple if everything is already defined in the model.
        return self.model(batch['image'])

    def prepare_data(self):
        # This is called at the start of training
        pass

    def train_dataloader(self):
        # Simply define a pytorch dataloader here that will take care of batching. Note it works well with dictionnaries !
        train_dl = tdata.DataLoader(self.trainset, batch_size=self.hparams.batch_size, shuffle=True,
                                    num_workers=os.cpu_count())
        return train_dl

    def val_dataloader(self):
        # Same but for validation. Pytorch lightning allows multiple validation dataloaders hence why I return a list.
        val_dl = tdata.DataLoader(self.valset, batch_size=self.hparams.batch_size, shuffle=False,
                                  num_workers=os.cpu_count()) 
        return [val_dl]
    
    def test_dataloader(self):
        test_dl = tdata.DataLoader(self.testset, batch_size=self.hparams.batch_size, shuffle=False,
                                  num_workers=os.cpu_count()) 
        return [test_dl]
    

    def loss_function(self, logits, gt):
        # How to calculate the loss. Note this method is actually not a part of pytorch lightning ! It's only good practice
        #loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([32542/584]).to(logits.device))  # Let's rebalance the weights for each class here.
        loss_fn = FocalLoss(logits=True)
        gt = gt.float()
        loss = loss_fn(logits, gt)
        return loss

    def configure_optimizers(self):
        # Optimizers and schedulers. Note that each are in lists of equal length to allow multiple optimizers (for GAN for example)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=3e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=10 * self.hparams.lr, 
                                                        epochs=self.hparams.epochs, steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # This is where you must define what happens during a training step (per batch)
        logits = self(batch)
        loss = self.loss_function(logits, batch['target']).unsqueeze(0)  # You need to unsqueeze in case you do multi-gpu training
        # Pytorch lightning will call .backward on what is called 'loss' in output
        # 'log' is reserved for tensorboard and will log everything define in the dictionary
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        # This is where you must define what happens during a validation step (per batch)
        logits = self(batch)
        loss = self.loss_function(logits, batch['target']).unsqueeze(0)
        probs = torch.sigmoid(logits)
        return {'val_loss': loss, 'probs': probs, 'gt': batch['target']}
    
    def test_step(self, batch, batch_idx):
        logits = self(batch)
        probs = torch.sigmoid(logits)
        return {'probs': probs}

    def validation_epoch_end(self, outputs):
        # This is what happens at the end of validation epoch. Usually gathering all predictions
        # outputs is a list of dictionary from each step.
        avg_loss = torch.cat([out['val_loss'] for out in outputs], dim=0).mean()
        probs = torch.cat([out['probs'] for out in outputs], dim=0)
        gt = torch.cat([out['gt'] for out in outputs], dim=0)
        probs = probs.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        auc_roc = torch.tensor(roc_auc_score(gt, probs))
        tensorboard_logs = {'val_loss': avg_loss, 'auc': auc_roc}
        print(f'Epoch {self.current_epoch}: {avg_loss:.2f}, auc: {auc_roc:.4f}')

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_epoch_end(self, outputs):
        probs = torch.cat([out['probs'] for out in outputs], dim=0)
        probs = probs.detach().cpu().numpy()
        self.test_predicts = probs  # Save prediction internally for easy access
        # We need to return something 
        return {'dummy_item': 0}


# 
# ## Training
# 
# Let's start by specifying parameters, the seed and output folder.
# 

# In[ ]:


# dict_to_args is a simple helper to make hparams act like args from argparse. This makes it trivial to then use argparse
OUTPUT_DIR = './lightning_logs'
hparams = dict_to_args({'batch_size': 64,
                        'lr': 1e-4, # common when using pretrained
                        'epochs': 10,
                        'arch': 'seresnext50'
                       })


# For training we just need to instantiate the pytorch lightning module and a trainer with a few options. Most importantly this is where you specify how many GPU to use (or TPU) and if you want to do mixed precision training (with apex). For the purpose of this kernel I just do FP32 1GPU training but please read the pytorch lightning doc if you want to try TPU and/or mixed precision.

# In[ ]:


# Initiate model
model = LightModel(train_df, test_df, pid_train, pid_val, hparams)
tb_logger = pl.loggers.TensorBoardLogger(save_dir='./',
                                         name=f'baseline', # This will create different subfolders for your models
                                         version=f'0')  # If you use KFold you can specify here the fold number like f'fold_{fold+1}'
checkpoint_callback = pl.callbacks.ModelCheckpoint(filename=tb_logger.log_dir + "/{epoch:02d}-{auc:.4f}",
                                                   monitor='auc', mode='max')
# Define trainer
# Here you can 
trainer = pl.Trainer(max_epochs=hparams.epochs, auto_lr_find=False,  # Usually the auto is pretty bad. You should instead plot and pick manually.
                     gradient_clip_val=1,
                     num_sanity_val_steps=0,  # Comment that out to reactivate sanity but the ROC will fail if the sample has only class 0
                     checkpoint_callback=checkpoint_callback,
                     gpus=1,
                     progress_bar_refresh_rate=0
                     )


# In[ ]:


trainer.fit(model)


# ## Test time
# The easy part :)

# In[ ]:


# Grab best checkpoint file
out = Path(tb_logger.log_dir)
aucs = [ckpt.stem[-4:] for ckpt in out.iterdir()]
best_auc_idx = aucs.index(max(aucs))
best_ckpt = list(out.iterdir())[best_auc_idx]
print('Using ', best_ckpt)


# In[ ]:


trainer = pl.Trainer(resume_from_checkpoint=str(best_ckpt), gpus=1)


# In[ ]:


trainer.test(model)


# In[ ]:


preds = model.test_predicts
test_df['target'] = preds
submission = test_df[['image_name', 'target']]
submission.to_csv('submission.csv', index=False)

