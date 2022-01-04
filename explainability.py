from torchvision import transforms
import torch
import os
from argparse import ArgumentParser 
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50
from melanoma_cnn_efficientnet import Net , CustomDataset
import json
import random
import cv2
import numpy as np
import pandas as pd

from captum.attr import GuidedGradCam, IntegratedGradients, GradientShap, Occlusion, NoiseTunnel
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

transform = transforms.ToTensor()
testing_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(256),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

directories = ["/workspace/stylegan2-ada-pytorch/processed_dataset_256"]
filename = "dataset.json"

input_images = [str(f) for f in sorted(directories[0].rglob('*')) if os.path.isfile(f)]
y = [1 for n in range(len(input_images))] #[0 if f.split('.jpg')[0][-1] == '0' else 1 for f in input_images]
data_df = pd.DataFrame({'image_name': input_images, 'target': y})
dataset = CustomDataset(df = data_df, train = False, transforms = testing_transforms )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
                 

# arch = resnet50(pretrained=True) 
arch = EfficientNet.from_pretrained('efficientnet-b2')
model = Net(arch=arch)  
# summary(model, (3, 256, 256), device='cpu')
model.load_state_dict(torch.load('/workspace/stylegan2-ada-pytorch/CNN_trainings/melanoma_model_0_0.9225_16_12_train_reals+15melanoma.pth')) #training_classifiers_events/12_29/melanoma_model_0_0.9818_2021-12-29-resnet.pth
model.eval().to(device)

guided_gc = GuidedGradCam(model, model.arch.layer4[2].conv3) #model.arch._conv_head )
integrated_gradients = IntegratedGradients(model)
occlusion = Occlusion(model)
gradient_shap = GradientShap(model)

# grad-cam library
target_layers = [model.arch._conv_head] #[model.arch.layer4[-1]]
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
rgb_img = cv2.imread(img_dir, 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255

                            
for directory in directories:
    with open(os.path.join(directory, filename)) as file:
        data = json.load(file)['labels']
        random.shuffle(data)
        for i, (img, label) in enumerate(data):
            img_dir = os.path.join(directory,img) 
            plot_diagnosis(img_dir, model, label)
            image = torch.tensor(testing_transforms(Image.open(img_dir)).unsqueeze(0), 
                                    dtype=torch.float32, requires_grad=True).to(device)
            
            # The input tensor can be a batch tensor with several images.    
            grayscale_cam = cam(image, aug_smooth=True, eigen_smooth=True)
            visualization = show_cam_on_image(rgb_img, grayscale_cam[0,:], use_rgb=True)
            cv2.imwrite(f'grad_cam.jpg', visualization)

            
            attr_guidedGC = guided_gc.attribute(image)
            attr_ig = integrated_gradients.attribute(image)

            transposed_attr_ig = np.transpose(attr_ig.squeeze().detach().cpu().numpy(), (1,2,0))
            transposed_attr_guidedGC = np.transpose(attr_guidedGC.squeeze().detach().cpu().numpy(), (1,2,0))
            transposed_image = np.transpose(transform(Image.open(img_dir)).numpy(), (1,2,0))

            default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)
            
            _ = viz.visualize_image_attr(transposed_attr_ig,
                             transposed_image,
                             method='heat_map',
                             #cmap=default_cmap,
                             show_colorbar=True,
                             sign='absolute_value',
                             outlier_perc=1)
            plt.savefig('/workspace/stylegan2-ada-pytorch/attr_ig.png')
            _ = viz.visualize_image_attr(transposed_attr_guidedGC,
                             transposed_image,
                             method='heat_map',
                             #cmap=default_cmap,
                             show_colorbar=True,
                             sign='absolute_value',
                             outlier_perc=1)
            plt.savefig('/workspace/stylegan2-ada-pytorch/attr_guidedGC_' + img_dir.split('/')[-1].split('.')[0] + '.png')

            # For a better visual of the attribution, the images between baseline and target are sampled 
            # using a noise tunnel (by adding gaussian noise). And when the gradients are calulcated, 
            # we smoothe them by calculating their mean squared.

            noise_tunnel = NoiseTunnel(integrated_gradients)

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
                                                ["absolute_value", "absolute_value"],
                                                #cmap=default_cmap,
                                                show_colorbar=True)
            plt.savefig('/workspace/stylegan2-ada-pytorch/attr_guidedgc_noisetunnel_' + img_dir.split('/')[-1].split('.')[0] + '.png')

            # Occlusion based attribution

            attr_occ = occlusion.attribute(image,
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

            attributions_gs = gradient_shap.attribute(image,
                                                    n_samples=50,
                                                    stdevs=0.0001,
                                                    baselines=rand_img_dist)

            _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                transposed_image,
                                                ["original_image", "heat_map"],
                                                ["all", "absolute_value"],
                                                cmap=default_cmap,
                                                show_colorbar=True)
            plt.savefig('/workspace/stylegan2-ada-pytorch/gradshap_' + img_dir.split('/')[-1].split('.')[0] + '.png')
            
