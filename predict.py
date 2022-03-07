# Last Modified   : 22.01.2022
# By              : Sandra Carrasco <sandra.carrasco@ai.se>

import re
import os
from typing import List
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser
from melanoma_classifier import test
from utils import (load_model, load_isic_data, predict,
                   process_image, imshow,
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
