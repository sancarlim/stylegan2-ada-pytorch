#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : server_mp.py
# Modified   : 22.01.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

import os
import random
import argparse
from pathlib import Path
import PIL.Image as Image
import pandas as pd
import numpy as np 
from argparse import ArgumentParser, Namespace
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
import torch.utils.data as tdata
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from torchvision import transforms

import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
import json

# Reproductibility
SEED = 33
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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
        image_fn = meta['image_name'] + '.jpg'
        if self.test:
            img = Image.open(str(IMAGE_DIR / ('test/test/' + image_fn)))
        else:
            img = Image.open(str(IMAGE_DIR / ('train_224/' + image_fn)))
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.test:
            return {'image': img}
        else:
            return {'image': img, 'target': meta['target']}



class Synth_Dataset(tdata.Dataset):
    
    def __init__(self, transform, test=True):
        self.transform = transform
        self.test = test 
        self.input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if os.path.isfile(f)]

    def __len__(self):
        return len(self.input_images)
        
    def __getitem__(self, idx): 
        #class 0 - bening , class_1 - malign
        image_fn = self.input_images[idx]   #f'{idx:04d}_{idx%2}'

        img = Image.open(os.path.join(source_dir,image_fn))
        target = int( int(image_fn.split('seed')[1].replace('.jpg','')) > 2500 )  #class 1 seeds=2501-5000
        
        if self.transform is not None:
            img = self.transform(img)

        return {'image': img, 'target':target}


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
        self.arch = arch
        if arch == 'resnet34':
            remove_range = 2
            m = models.resnet34(pretrained=True)
            #fc = nn.Linear(in_features=512, out_features=500, bias=True)
        elif arch == 'efficientnet': 
            m = EfficientNet.from_pretrained("efficientnet-b6")
            m._fc = nn.Linear(in_features=2304, out_features=500, bias=True)
            self.base = m
            self.head = nn.Linear(500, 1)
            #fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        elif arch == 'seresnext50':
            m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
            #fc = nn.Linear(in_features=512, out_features=500, bias=True)
            remove_range = 2
            
        if arch != 'efficientnet':
            c_feature = list(m.children())[-1].in_features
            self.base = nn.Sequential(*list(m.children())[:-remove_range])
            self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(c_feature * 2, c_out))

        
    def forward(self, x):
        h = self.base(x)
        logits = self.head(h).squeeze(1)  
        return logits


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

 
class LightModel(pl.LightningModule):

    def __init__(self, df_train, df_test, pid_train, pid_val, test_syn=False):
        # This is where paths and options should be stored. I also store the
        # train_idx, val_idx for cross validation since the dataset are defined 
        # in the module !
        super().__init__()
        self.pid_train = pid_train
        self.pid_val = pid_val
        self.df_train = df_train

        self.model = Model(arch=hparams.arch)  # You will obviously want to make the model better :)

        
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
        
        if test_syn:
            self.testset = Synth_Dataset(self.transform_test, test=True)
        else:
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
        train_dl = tdata.DataLoader(self.trainset, batch_size=hparams.batch_size, shuffle=True,
                                    num_workers=os.cpu_count())
        return train_dl

    def val_dataloader(self):
        # Same but for validation. Pytorch lightning allows multiple validation dataloaders hence why I return a list.
        val_dl = tdata.DataLoader(self.valset, batch_size=hparams.batch_size, shuffle=False,
                                  num_workers=os.cpu_count()) 
        return [val_dl]
    
    def test_dataloader(self):
        test_dl = tdata.DataLoader(self.testset, batch_size=hparams.batch_size, shuffle=False,
                                  num_workers=os.cpu_count()) 
        return [test_dl]
    

    def loss_function(self, logits, gt):
        # How to calculate the loss. Note this method is actually not a part of pytorch lightning ! It's only good practice
        if self.loss == 'bce':
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([32542/584]).to(logits.device))  # Let's rebalance the weights for each class here.
        elif self.loss == 'focal':
            loss_fn = FocalLoss(logits=True)
            gt = gt.float()
            loss = loss_fn(logits, gt)
        return loss

    def configure_optimizers(self):
        # Optimizers and schedulers. Note that each are in lists of equal length to allow multiple optimizers (for GAN for example)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.lr, weight_decay=3e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=10 * hparams.lr, 
                                                        epochs=hparams.epochs, steps_per_epoch=len(self.train_dataloader()))
                    # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)
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

        predicted = torch.round(probs)
        loss = self.loss_function(logits, batch['target']).unsqueeze(0) 
        return {'prediction': predicted, 'probs': probs, 'gt': batch['target']}

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
        gt = torch.cat([out['gt'] for out in outputs], dim=0)
        predicted = torch.cat([out['predicted'] for out in outputs], dim=0)
        probs = probs.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        predicted = predicted.detach().cpu().numpy()

        total = gt.size(0)
        correct = (predicted == gt).sum().item()

        print('Accuracy of the network on the test images: %d %%' % ( 100 * correct / total))

        self.test_predicts = probs  # Save prediction internally for easy access

        auc_roc = torch.tensor(roc_auc_score(gt, probs))
        print(f'auc: {auc_roc:.4f}')
        # We need to return something 
        return {'acc': ( 100 * correct / total), 'auc': auc_roc}


training_dir = Path('/home/Data/melanoma_external_256')
train_df = pd.read_csv(training_dir/'train.csv')
test_synth_dir = Path('./generated')
test_df = pd.read_csv(training_dir/'test.csv') 
IMAGE_DIR = Path(training_dir)
source_dir = '/home/stylegan2-ada-pytorch/generated'
#frames=[train_df, test_df]
#joint_df = pd.concat(frames) 

""" 
labels_list = []
for n in range(len(train_df)):
    labels_list.append([train_df.iloc[n].image_name,int(train_df.iloc[n].target)])
labels_list_dict = { "labels" : labels_list}

with open("labels.json", "w") as outfile:
    json.dump(labels_list_dict, outfile) """


# So you have patients that have multiple images. Also apparently the data is imbalanced. Let's verify:
#train_df.groupby(['target']).count()
# so we have approx 60 times more negatives than positives. We need to make sure we split good/bad patients equally.

#df = pd.read_csv('/kaggle/input/melanoma-external-malignant-256/train_concat.csv')
patient_means = train_df.groupby(['patient_id'])['target'].mean()
patient_ids = train_df['patient_id'].unique()

# Now let's make our split
train_idx, val_idx = train_test_split(np.arange(len(patient_ids)), stratify=(patient_means > 0), test_size=0.2)  # KFold + averaging should be much better considering how small the dataset is for malignant cases
                        #train_test_split(df, stratify=df.target, test_size = 0.2, random_state=42)    
                    
pid_train = patient_ids[train_idx]
pid_val = patient_ids[val_idx]

# dict_to_args is a simple helper to make args act like args from argparse. This makes it trivial to then use argparse
OUTPUT_DIR = './lightning_logs'


# For training we just need to instantiate the pytorch lightning module and a trainer with a few options. Most importantly this is where you specify how many GPU to use (or TPU) and if you want to do mixed precision training (with apex). For the purpose of this kernel I just do FP32 1GPU training but please read the pytorch lightning doc if you want to try TPU and/or mixed precision.



def main(args: Namespace):
    tb_logger = pl.loggers.TensorBoardLogger(save_dir='./',
                                            name=f'baseline', # This will create different subfolders for your models
                                            version=f'0')  # If you use KFold you can specify here the fold number like f'fold_{fold+1}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename=tb_logger.log_dir + "/{epoch:02d}-{auc:.4f}",
                                                    monitor='auc', mode='max')
    # Define trainer 
    trainer = pl.Trainer(max_epochs=args.epochs, auto_lr_find=False,  # Usually the auto is pretty bad. You should instead plot and pick manually.
                        gradient_clip_val=1,
                        num_sanity_val_steps=0,  # Comment that out to reactivate sanity but the ROC will fail if the sample has only class 0
                        checkpoint_callback=checkpoint_callback,
                        gpus=1,
                        progress_bar_refresh_rate=0
                        )

    if args.test:
        model = LightModel(train_df, test_df, pid_train, pid_val, test_syn=True)
        trainer = pl.Trainer(resume_from_checkpoint=args.ckpt, gpus=1)
        trainer.test(model)

    else:
        model = LightModel(train_df, test_df, pid_train, pid_val)
        print('TRAINING STARTING...')
        trainer.fit(model)
        print('TRAINING FINISHED')

        # Grab best checkpoint file
        out = Path(tb_logger.log_dir)
        aucs = [ckpt.stem[-4:] for ckpt in out.iterdir()]
        best_auc_idx = aucs.index(max(aucs))
        best_ckpt = list(out.iterdir())[best_auc_idx]
        print('TEST: Using ', best_ckpt) 
        trainer = pl.Trainer(resume_from_checkpoint=best_ckpt, gpus=1)
        
        trainer.test(model)


        preds = model.test_predicts
        test_df['target'] = preds
        submission = test_df[['image_name', 'target']]
        submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help='Number of training epochs')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test", action='store_true', help='Only testing')
    parser.add_argument("--arch", type=str, default='efficientnet', help='Choose architecture',
                            choices=['seresnext50', 'resnet34', 'efficientnet' ])
    parser.add_argument("--loss", type=str, default='focal', help='Choose loss function',
                            choices=['bce', 'focal'])
    parser.add_argument("--ckpt", type=str, default='./training_runs/00000--cond-mirror-auto2/epoch=09-auc=0.8981.ckpt', help='CKPT path for testing')
    hparams = parser.parse_args()

    main(hparams)