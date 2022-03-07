# Last Modified   : 22.01.2022
# By              : Sandra Carrasco <sandra.carrasco@ai.se>

import os
import PIL.Image as Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from argparse import ArgumentParser
import sys
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from utils import Net, testing_transforms
from pathlib import Path


def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--use_cnn",
                        help=f'retrieve features from'
                             f'the last layer of EfficientNet B2',
                        action='store_true')
    parser.add_argument("--sprite", action='store_true')
    parser.add_argument("--model_path", type=str,
                        help='path to trained classifier EfficientNet-B2')
    parser.add_argument("--projections_path", type=str,
                        help='path to generated projections')
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help='path to save embeddings')
    args = parser.parse_args()

    # Setting up GPU for processing or CPU if GPU isn't available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()

    if args.use_cnn:
        if args.embeddings_path is None:
            sys.exit("You mast provide embeddings path!")

        arch = EfficientNet.from_pretrained('efficientnet-b2')
        model = Net(arch=arch, return_feats=True)
        model.load_state_dict(torch.load(args.model_path))

        model.eval()
        model.to(device)
        images_pil = []
        metadata_f = []
        embeddings = []

        # Repeat the process for randomly generated data
        images = [
            str(f) for f in sorted(
                Path(args.projections_path).glob('*png')
                ) if os.path.isfile(f)]
        labels = []
        for f in images:
            if "from" in f:
                labels.append(f.split('.from.png')[0][-1])
            else:
                labels.append(str(int(f.split('.to.png')[0][-1]) + 2))

        with torch.no_grad():
            for img_dir, label in tqdm(zip(images, labels)):
                img_net = torch.tensor(testing_transforms(
                    Image.open(img_dir)).unsqueeze(0),
                    dtype=torch.float32).to(device)
                emb = model(img_net)
                embeddings.append(emb.cpu())
                metadata_f.append([label, img_dir.split('/')[-1]])
                if args.sprite:
                    img_pil = transform(Image.open(img_dir).resize((100, 100)))
                    images_pil.append(img_pil)

        embeddings_tensor = torch.stack(embeddings).squeeze()
        if args.sprite:
            images_pil = torch.stack(images_pil)
        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter(args.embeddings_path)
    else:
        # This part can be used with G_mapping embeddings (vector w)
        # - projections in the latent space
        directory = args.projections_path
        emb_f = "allvectorsf.txt"
        metadata_f = "alllabelsf.txt"
        transform = transforms.ToTensor()

        with open(os.path.join(directory, emb_f)) as f:
            embeddings = f.readlines()  # [::2]
        embeddings_tensor = torch.tensor(
            [float(i) for emb_line in embeddings for i in emb_line[
                :-2].split(' ')]
            ).reshape(len(embeddings), -1)

        with open(os.path.join(directory, metadata_f)) as f:
            metadata = f.readlines()  # [::2]
        metadata_f = [
            [
                name.split('.')[0].split(
                    ' ')[0], name.split('.')[0].split(' ')[1]
            ] for name in metadata
            ]

        images_pil = torch.empty(len(metadata), 3, 100, 100)
        labels = []
        for i, line in enumerate(metadata):
            label = int(line.split(' ')[0])
            if label == 0 or label == 1:
                img_name = '00000/'
                img_name += line.split(' ')[1].split('txt')[0] + 'from.png'
            else:
                label_name = '0' if label == 2 else '1'
                img_name = 'generated-20kpkl/'
                img_name += line.split(' ')[1].split('.')[0]
                img_name += '_' + label_name + '.jpg'

            img_dir = os.path.join(directory, img_name)
            img = transform(Image.open(img_dir).resize((100, 100)))
            images_pil[i] = img
            labels.append(label)

        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter(
            args.projections_path + directory.split('/')[-1])

    if args.sprite:
        writer.add_embedding(embeddings_tensor,
                             metadata=metadata_f,
                             metadata_header=["label", "image_name"],
                             label_img=images_pil)
    else:
        writer.add_embedding(embeddings_tensor,
                             metadata=metadata_f,
                             metadata_header=["label", "image_name"])
    writer.close()
