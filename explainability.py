from torchvision import transforms
import torch
import os
from argparse import ArgumentParser 
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50
from melanoma_cnn_efficientnet import Net , CustomDataset, train_test_split
import json
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from captum.attr import GuidedGradCam, IntegratedGradients, GradientShap, Occlusion, NoiseTunnel, Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from predict import plot_diagnosis

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# from fastai2.vision.all import *


torch.manual_seed(0)
np.random.seed(0)

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)

transform = transforms.ToTensor()
testing_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(256),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

directories = ["/workspace/stylegan2-ada-pytorch/processed_dataset_256_SAM","/workspace/stylegan2-ada-pytorch/processed_dataset_256"]
filename = "dataset.json"

# input_images = [str(f) for f in sorted(Path(directories[0]).rglob('*')) if os.path.isfile(f)]
# y = [1 for n in range(len(input_images))] #[0 if f.split('.jpg')[0][-1] == '0' else 1 for f in input_images]
# data_df = pd.DataFrame({'image_name': input_images, 'target': y})

# For testing with ISIC dataset
""" df = pd.read_csv(os.path.join('/workspace/melanoma_isic_dataset' , 'train_concat.csv')) 
train_img_dir = os.path.join('/workspace/melanoma_isic_dataset' ,'train/train/')
train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42) 
validation_df=pd.DataFrame(valid_split)
validation_df['image_name'] = [os.path.join(train_img_dir, validation_df.iloc[index]['image_name'] + '.jpg') for index in range(len(validation_df))]
dataset = CustomDataset(df = validation_df, train = False, transforms = testing_transforms )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
"""
arch_r = resnet50(pretrained=True) 
arch_ef = EfficientNet.from_pretrained('efficientnet-b2')
model_r = Net(arch=arch_r)  
model_ef = Net(arch=arch_ef)  
# summary(model, (3, 256, 256), device='cpu')
model_ef.load_state_dict(torch.load('/workspace/stylegan2-ada-pytorch/CNN_trainings/melanoma_model_0_0.9225_16_12_train_reals+15melanoma.pth')) 
model_r.load_state_dict(torch.load('/workspace/stylegan2-ada-pytorch/training_classifiers_events/12_29/melanoma_model_0_0.9818_2021-12-29-resnet.pth'))

model_r.eval().to(device)
model_ef.eval().to(device)

guided_gc_ef = GuidedGradCam(model_ef,model_ef.arch._conv_head) 
guided_gc_r = GuidedGradCam(model_r, model_r.arch.layer4[-1]) 
integrated_gradients_ef = IntegratedGradients(model_ef)
integrated_gradients_r = IntegratedGradients(model_r)
saliency_ef = Saliency(model_ef)
saliency_r = Saliency(model_r)
occlusion_ef = Occlusion(model_ef)
occlusion_r = Occlusion(model_r)
gradient_shap_ef = GradientShap(model_ef)
gradient_shap_r = GradientShap(model_r)

# grad-cam library
target_layers_ef = [model_ef.arch._conv_head]
target_layers_r = [model_r.arch.layer4[-1]]
# Construct the CAM object once, and then re-use it on many images:
cam_ef = EigenCAM(model=model_ef, target_layers=target_layers_ef, use_cuda=True)
cam_r = EigenCAM(model=model_r, target_layers=target_layers_r, use_cuda=True)
                            
for directory in directories:
    with open(os.path.join(directory, filename)) as file:
        data = json.load(file)['labels']
        random.shuffle(data)
        for i, (img, label) in enumerate(data):
            img_dir = os.path.join(directory,img) 
            # for img_dir, image, label in dataloader:
            #     img_dir=img_dir[0]
            #     image=image.to(device)
            image = torch.tensor(testing_transforms(Image.open(img_dir)).unsqueeze(0), 
                                    dtype=torch.float32).to(device)

            pred_resnet = torch.sigmoid(model_r(image))
            pred_effnet = torch.sigmoid(model_ef(image))
            print(pred_resnet, pred_effnet, label)
            #plot_diagnosis(img_dir, model_ef, label)
            
            transposed_image = np.transpose(transform(Image.open(img_dir)).numpy(), (1,2,0))
            rgb_img = cv2.imread(img_dir, 1)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            # The input tensor can be a batch tensor with several images.    
            grayscale_cam_ef = cam_ef(image, aug_smooth=True, eigen_smooth=True)
            grayscale_cam_r = cam_r(image, aug_smooth=True, eigen_smooth=True)
            visualization = show_cam_on_image(rgb_img, grayscale_cam_ef[0,:], use_rgb=True)
            cv2.imwrite(f'/workspace/stylegan2-ada-pytorch/grad_cam_ef'+ img_dir.split('/')[-1].split('.')[0] + '.jpg', visualization)
            visualization = show_cam_on_image(rgb_img, grayscale_cam_r[0,:], use_rgb=True)
            cv2.imwrite(f'/workspace/stylegan2-ada-pytorch/grad_cam_r'+ img_dir.split('/')[-1].split('.')[0] + '.jpg', visualization)


            # Occlusion based attribution

            attr_occ = occlusion_r.attribute(image,
                                            strides = (3, 8, 8), 
                                            sliding_window_shapes=(3,15, 15),
                                            baselines=0)

            _ = viz.visualize_image_attr_multiple(np.transpose(attr_occ.squeeze().detach().cpu().numpy(), (1,2,0)),
                                        transposed_image,
                                        ["original_image", "blended_heat_map"],
                                        ["all", "all"],
                                        show_colorbar=True,
                                        outlier_perc=2,
                                        )

            plt.savefig('/workspace/stylegan2-ada-pytorch/occlusion_' + img_dir.split('/')[-1].split('.')[0] + '.png')
            
            # Defining baseline distribution of images
            rand_img_dist = torch.cat([image * 0, image * 1])

            attributions_gs = gradient_shap_ef.attribute(image,
                                                    n_samples=50,
                                                    stdevs=0.0001,
                                                    baselines=rand_img_dist)

            _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                transposed_image,
                                                ["original_image", "heat_map"],
                                                ["all", "absolute_value"],
                                                #cmap=default_cmap,
                                                show_colorbar=True)
            plt.savefig('/workspace/stylegan2-ada-pytorch/gradshap_' + img_dir.split('/')[-1].split('.')[0] + '.png')

            attr_saliency = saliency_ef.attribute(image)
            attr_guidedGC = guided_gc_ef.attribute(image)
            attr_ig = integrated_gradients_ef.attribute(image)

            transposed_attr_ig = np.transpose(attr_ig.squeeze().detach().cpu().numpy(), (1,2,0))
            transposed_attr_guidedGC = np.transpose(attr_guidedGC.squeeze().detach().cpu().numpy(), (1,2,0))
            transposed_attr_saliency = np.transpose(attr_saliency.squeeze().detach().cpu().numpy(), (1,2,0))
            
            _ = viz.visualize_image_attr(transposed_attr_ig,
                                transposed_image,
                                method='heat_map',
                                #cmap=default_cmap,
                                show_colorbar=True,
                                sign='all',
                                outlier_perc=1)
            plt.savefig('/workspace/stylegan2-ada-pytorch/attr_ig.png')
            # _ = viz.visualize_image_attr(transposed_attr_guidedGC,
            #                     transposed_image,
            #                     method='heat_map',
            #                     #cmap=default_cmap,
            #                     show_colorbar=True,
            #                     sign='absolute_value',
            #                     outlier_perc=1)
            # plt.savefig('/workspace/stylegan2-ada-pytorch/attr_guidedGC_' + img_dir.split('/')[-1].split('.')[0] + '.png')
            _ = viz.visualize_image_attr(transposed_attr_saliency,
                                transposed_image,
                                method='heat_map',
                                #cmap=default_cmap,
                                show_colorbar=True,
                                sign='all',
                                outlier_perc=1)
            plt.savefig('/workspace/stylegan2-ada-pytorch/saliency.png')
            # For a better visual of the attribution, the images between baseline and target are sampled 
            # using a noise tunnel (by adding gaussian noise). And when the gradients are calulcated, 
            # we smoothe them by calculating their mean squared.

            """ noise_tunnel = NoiseTunnel(saliency_ef)

            attributions_ig_nt = noise_tunnel.attribute(image, nt_samples=5, nt_type='smoothgrad')
            transposed_attr_ig_nt = np.transpose(attributions_ig_nt.squeeze().detach().cpu().numpy(), (1,2,0))
            _ = viz.visualize_image_attr_multiple(transposed_attr_ig_nt,
                                                transposed_image,
                                                ["original_image", "heat_map"],
                                                ["all", "positive"],
                                                #cmap=default_cmap,
                                                show_colorbar=True)
            plt.savefig('/workspace/stylegan2-ada-pytorch/attr_ig_noisetunnel_' + img_dir.split('/')[-1].split('.')[0] + '.png')

            noise_tunnel = NoiseTunnel(guided_gc)

            attributions_guidedgc_nt = noise_tunnel.attribute(image, nt_samples=5, nt_type='smoothgrad')
            transposed_attr_guidedgc_nt = np.transpose(attributions_guidedgc_nt.squeeze().detach().cpu().numpy(), (1,2,0))
            _ = viz.visualize_image_attr_multiple(transposed_attr_guidedgc_nt,
                                                transposed_image,
                                                ["original_image", "heat_map"],
                                                ["all", "all"],
                                                #cmap=default_cmap,
                                                show_colorbar=True)
            plt.savefig('/workspace/stylegan2-ada-pytorch/attr_guidedgc_noisetunnel_' + img_dir.split('/')[-1].split('.')[0] + '.png')
            """