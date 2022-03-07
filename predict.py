# Last Modified   : 22.01.2022
# By              : Sandra Carrasco <sandra.carrasco@ai.se>

import numpy as np
import re
import os
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
import torch
from argparse import ArgumentParser
from melanoma_classifier import test
from utils import (load_model, load_isic_data,
                   load_synthetic_data,  CustomDataset,
                   confussion_matrix, testing_transforms)


def num_range(s: str) -> List[int]:
    '''
    Accept either a comma separated list of numbers
    'a,b,c' or a range 'a-c' and return as a list of ints.
    '''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def process_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model

    pil_image = Image.open(image_path)

    # Resize
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))

    # Crop
    left_margin = (pil_image.width-256)/2
    bottom_margin = (pil_image.height-256)/2
    right_margin = left_margin + 256
    top_margin = bottom_margin + 256

    pil_image = pil_image.crop((left_margin, bottom_margin,
                                right_margin, top_margin))

    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # PyTorch expects the color channel to be the first dimension
    # but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other
    # two dimensions.
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    if title is not None:
        ax.set_title(title)

    # Image needs to be clipped between 0 and 1
    # or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=1, prob=0.5):
    # just 2 classes from 1 single output
    '''
    Predict the class (or classes) of an image
    using a trained deep learning model.
    '''
    output = model(testing_transforms(
        Image.open(image_path)).type(
            torch.cuda.FloatTensor).unsqueeze(0))  # same output

    probabilities = torch.sigmoid(output)

    # Probabilities and the indices of those probabilities
    # corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)

    # Convert to lists
    top_probabilities = top_probabilities.detach().type(
        torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(
        torch.FloatTensor).numpy().tolist()[0]

    top_classes = []

    if probabilities > prob:
        top_classes.append("Melanoma")
    else:
        top_classes.append("Benign")

    return top_probabilities, top_classes


def plot_diagnosis(predict_image_path, model, label):
    img_nb = predict_image_path.split('/')[-1].split('.')[0]
    probs, classes = predict(predict_image_path, model)

    # Display an image along with the diagnosis of melanoma or benign
    # Plot Skin image input image
    plt.figure(figsize=(6, 10))
    plot_1 = plt.subplot(2, 1, 1)

    image = process_image(predict_image_path)

    imshow(image, plot_1)
    if (('Benign' in classes and label == 0)
       or ('Melanoma' in classes and label == 1)):
        font = {"color": 'g'}
    else:
        font = {"color": 'r'}
    plot_1.set_title(
        f"Diagnosis: {classes}, Output (prob) {probs[0]:.4f}, Label: {label}",
        fontdict=font)
    plt.savefig(f'{args.out_path}/prediction_{img_nb}.png')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seeds', type=num_range,
                        help='List of random seeds Ex. 0-3 or 0,1,2')
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--out_path", type=str, default='',
                        help='output path for confussion matrix')
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Plot and save image with diagnosis",
    )
    args = parser.parse_args()

    # Setting up GPU for processing or CPU if GPU isn't available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    if "isic" in args.data_path:
        # For testing with ISIC dataset
        _, test_df = load_isic_data(args.data_path)
    else:
        test_df = load_synthetic_data(args.data_path, "3,3")

    testing_dataset = CustomDataset(df=test_df, train=True,
                                    transforms=testing_transforms)
    test_loader = torch.utils.data.DataLoader(testing_dataset,
                                              batch_size=16,
                                              shuffle=False)
    test_pred, test_gt, test_accuracy = test(model, test_loader)
    confussion_matrix(test_gt, test_pred, test_accuracy, args.out_path)

    # Plot diagnosis
    if args.plot:
        for seed_idx, seed in enumerate(args.seeds):
            print(
                f'Predicting image for seed '
                f'{seed} ({seed_idx}/{len(args.seeds)}) ...')
            path = os.path.join(args.out_path, 'seed' + str(seed).zfill(4))
            path += '_0.png' if seed <= 5000 else '_1.png'
            plot_diagnosis(path, model)
