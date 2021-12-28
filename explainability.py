from torchvision import transforms
import torch
import os
from argparse import ArgumentParser 
from PIL import Image
from efficientnet_pytorch import EfficientNet
from melanoma_cnn_efficientnet import Net 
import json
import random
import numpy as np

from captum.attr import GuidedGradCam, IntegratedGradients, GradientShap, Occlusion, NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# from fastai2.vision.all import *


# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
testing_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(256),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

directories = ["/workspace/stylegan2-ada-pytorch/processed_dataset_256_SAM", "/workspace/stylegan2-ada-pytorch/processed_dataset_256"]
filename = "dataset.json"

arch = EfficientNet.from_pretrained('efficientnet-b2')
model = Net(arch=arch)  
# summary(model, (3, 256, 256), device='cpu')
model.load_state_dict(torch.load('/workspace/stylegan2-ada-pytorch/CNN_trainings/melanoma_model_0_0.9225_16_12_train_reals+15melanoma.pth'))
model.to(device)

guided_gc = GuidedGradCam(model, model.arch._conv_head )
integrated_gradients = IntegratedGradients(model)
occlusion = Occlusion(model)

for directory in directories:
    with open(os.path.join(directory, filename)) as file:
        data = json.load(file)['labels']
        random.shuffle(data)
        for i, (img, label) in enumerate(data):
            img_dir = "/workspace/stylegan2-ada-pytorch/projector/generated-20kpkl7/seed0789_1.jpg" # os.path.join(directory,img) 
            image = torch.tensor(testing_transforms(Image.open(img_dir)).unsqueeze(0), 
                                    dtype=torch.float32, requires_grad=True).to(device)
            attr_guidedGC = guided_gc.attribute(image)
            attr_ig = integrated_gradients.attribute(image, n_steps=200)

            transposed_attr_ig = np.transpose(attr_ig.squeeze().detach().cpu().numpy(), (1,2,0))
            transposed_attr_guidedGC = np.transpose(attr_guidedGC.squeeze().detach().cpu().numpy(), (1,2,0))
            transposed_image = np.transpose(transform(Image.open(img_dir)).numpy(), (1,2,0))

            default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)
            
            _ = viz.visualize_image_attr(transposed_attr_ig,
                             transposed_image,
                             method='blended_heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='all',
                             outlier_perc=1)
            plt.savefig('/workspace/stylegan2-ada-pytorch/attr_ig.png')
            _ = viz.visualize_image_attr(transposed_attr_guidedGC,
                             transposed_image,
                             method='blended_heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='all',
                             outlier_perc=1)
            plt.savefig('/workspace/stylegan2-ada-pytorch/attr_guidedGC.png')

            # For a better visual of the attribution, the images between baseline and target are sampled 
            # using a noise tunnel (by adding gaussian noise). And when the gradients are calulcated, 
            # we smoothe them by calculating their mean squared.

            noise_tunnel = NoiseTunnel(integrated_gradients)

            attributions_ig_nt = noise_tunnel.attribute(image, nt_samples=5, nt_type='smoothgrad_sq')
            transposed_attr_ig_nt = np.transpose(attributions_ig_nt.squeeze().detach().cpu().numpy(), (1,2,0))
            _ = viz.visualize_image_attr_multiple(transposed_attr_ig_nt,
                                                transposed_image,
                                                ["original_image", "heat_map"],
                                                ["all", "all"],
                                                cmap=default_cmap,
                                                show_colorbar=True)
            plt.savefig('/workspace/stylegan2-ada-pytorch/attr_ig_noisetunnel.png')

            noise_tunnel = NoiseTunnel(guided_gc)

            attributions_guidedgc_nt = noise_tunnel.attribute(image, nt_samples=5, nt_type='smoothgrad_sq')
            transposed_attr_guidedgc_nt = np.transpose(attributions_guidedgc_nt.squeeze().detach().cpu().numpy(), (1,2,0))
            _ = viz.visualize_image_attr_multiple(transposed_attr_guidedgc_nt,
                                                transposed_image,
                                                ["original_image", "heat_map"],
                                                ["all", "all"],
                                                cmap=default_cmap,
                                                show_colorbar=True)
            plt.savefig('/workspace/stylegan2-ada-pytorch/attr_guidedgc_noisetunnel.png')

            # Occlusion based attribution

            attr_occ = occlusion.attribute(image,
                                            strides = (3, 8, 8), 
                                            sliding_window_shapes=(3,15, 15),
                                            baselines=0)

            _ = viz.visualize_image_attr_multiple(np.transpose(attr_occ.squeeze().detach().cpu().numpy(), (1,2,0)),
                                      transposed_image,
                                      ["original_image", "heat_map"],
                                      ["all", "all"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )

            plt.savefig('/workspace/stylegan2-ada-pytorch/occlusion.png')

