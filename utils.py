import numpy as np
import os
import cv2
import PIL.Image as Image
from matplotlib import pylab as P
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
import torchvision
import pandas as pd
import seaborn as sb
import datetime
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score 

from pathlib import Path
import random

from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

classes = ('benign', 'melanoma')

# Defining transforms for the training, validation, and testing sets
training_transforms = torchvision.transforms.Compose(
    [torchvision.transforms.RandomRotation(30),
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomVerticalFlip(),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])

validation_transforms = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(256),
     torchvision.transforms.CenterCrop(256),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])

testing_transforms = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(256),
     torchvision.transforms.CenterCrop(256),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Creating seeds to make results reproducible
def seed_everything(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

    pil_image = pil_image.crop(
        (left_margin, bottom_margin, right_margin, top_margin))

    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # PyTorch expects the color channel to be the
    # first dimension but it's the third dimension
    # in the PIL image and Numpy array
    # Color channel needs to be first;
    # retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def imshow(image, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots()

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


def predict(image_path, model, topk=1):
    # just 2 classes from 1 single output
    '''
    Predict the class (or classes) of an image using
    a trained deep learning model.
    '''
    image = process_image(image_path)

    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)

    # Returns a new tensor with a dimension of size
    # one inserted at the specified position.
    image = image.unsqueeze(0)
    output = model(image)
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

    if probabilities > 0.5:
        top_classes.append("Melanoma")
    else:
        top_classes.append("Benign")

    return top_probabilities, top_classes


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [
        F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def renormalize(tensor):
    minFrom = tensor.min()
    maxFrom = tensor.max()
    minTo = 0
    maxTo = 1
    return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = renormalize(img).cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(48, 48))
    for idx in np.arange(32):
        ax = fig.add_subplot(4, 8, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=(
                        "green" if preds[idx] == labels[idx].item(
                        ) else "red"))
    return fig


def add_pr_curve_tensorboard(class_index, test_probs,
                             test_label, writer, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_truth = test_label == class_index
    if class_index == 0:
        tensorboard_probs = 1 - test_probs
    else:
        tensorboard_probs = test_probs

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()


def confussion_matrix(test, test_pred, test_accuracy, writer_path):
    pred = np.round(test_pred)
    cm = confusion_matrix(test, pred)

    cm_df = pd.DataFrame(cm,
                         index=['Benign', 'Malignant'],
                         columns=['Benign', 'Malignant'])

    plt.figure(figsize=(5.5, 4))
    sb.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix \nAccuracy:{0:.3f}'.format(test_accuracy))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    now = datetime.datetime.now()
    plt.savefig(os.path.join(
        writer_path,
        f'conf_matrix_{test_accuracy:.4f}_{now.strftime("%d_%m_%H_%M")}.png'))


def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title)


def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)


def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)


def LoadImage(file_path):
    im = Image.open(file_path)
    im = np.asarray(im)
    return im


def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.tensor(images, dtype=torch.float32)
    images = transformer.forward(images).to('cuda')
    return images.requires_grad_(True)


def load_isic_data(path):
    # ISIC dataset
    df = pd.read_csv(os.path.join(path, 'train_concat.csv'))
    train_img_dir = os.path.join(path, 'train/train/')

    df['image_name'] = [
        os.path.join(
            train_img_dir, df.iloc[index]['image_name'] + '.jpg'
            ) for index in range(len(df))]

    train_split, valid_split = train_test_split(
        df, stratify=df.target, test_size=0.20, random_state=42)
    train_df = pd.DataFrame(train_split)
    validation_df = pd.DataFrame(valid_split)
    return train_df, validation_df


def load_synthetic_data(syn_data_path, synt_n_imgs, only_syn=False):
    # Load all images and labels from path
    input_images = [
        str(f) for f in sorted(
            Path(syn_data_path).rglob('*')) if os.path.isfile(f)]
    y = [0 if f.split('.jpg')[0][-1] == '0' else 1 for f in input_images]

    ind_0, ind_1 = [], []
    for i, _ in enumerate(input_images):
        if y[i] == 0:
            ind_0.append(i)
        else:
            ind_1.append(i)

    # Select number of melanomas and benign samples
    n_b, n_m = [
        float(i) for i in synt_n_imgs.split(',')
        ] if not only_syn else [1000, 1000]
    ind_0 = np.random.permutation(ind_0)[:int(n_b * 1000)]
    ind_1 = np.random.permutation(ind_1)[:int(n_m * 1000)]

    id_list = np.append(ind_0, ind_1)

    train_img = [input_images[int(i)] for i in id_list]
    train_gt = [y[int(i)] for i in id_list]
    train_df = pd.DataFrame({'image_name': train_img, 'target': train_gt})

    return train_df


def create_split(source_dir):
    # Split synthetic dataset
    input_images = [
        str(f) for f in sorted(
            Path(source_dir).rglob('*')) if os.path.isfile(f)]

    ind_0, ind_1 = [], []
    for i, f in enumerate(input_images):
        if f.split('.')[0][-1] == '0':
            ind_0.append(i)
        else:
            ind_1.append(i)

    train_id_list = ind_0[round(len(ind_0)*0.8):]
    val_id_list = ind_0[round(len(ind_0)*0.8):]
    train_id_1 = ind_1[:round(len(ind_1)*0.8)]
    val_id_1 = ind_1[round(len(ind_1)*0.8):]

    train_id_list = np.append(train_id_list, train_id_1)
    val_id_list = np.append(val_id_list, val_id_1)
    return train_id_list, val_id_list


class CustomDataset(Dataset):

    def __init__(self, df: pd.DataFrame,
                 train: bool = True, transforms=None):
        self.df = df
        self.transforms = transforms
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['image_name']
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        images = Image.open(img_path)

        if self.transforms:
            images = self.transforms(images)

        labels = self.df.iloc[index]['target']

        if self.train:
            # return images, labels
            return torch.tensor(
                images, dtype=torch.float32), torch.tensor(
                    labels, dtype=torch.float32)
        else:
            # return (images)
            return img_path, torch.tensor(
                images, dtype=torch.float32), torch.tensor(
                    labels, dtype=torch.float32)


class Synth_Dataset(Dataset):
    def __init__(self, source_dir, transform, id_list=None,
                 input_img=None, test=False, unbalanced=False):
        self.transform = transform
        self.source_dir = source_dir

        if input_img is None:
            self.input_images = [str(f) for f in sorted(
                Path(source_dir).rglob('*')) if os.path.isfile(f)]
        else:
            self.input_images = input_img

        if unbalanced:
            ind_0, ind_1 = create_split(source_dir)
            ind = np.append(ind_0, ind_1)
            self.input_images = [self.input_images[i] for i in ind]

        self.id_list = id_list if id_list is not None else range(
            len(self.input_images))

        if test:
            if unbalanced:
                self.input_images = self.input_images[:5954]

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        idx = self.id_list[idx]

        image_fn = self.input_images[idx]

        img = np.array(Image.open(image_fn))
        target = int(image_fn.split('_')[-1].replace('.jpg', ''))

        if self.transform is not None:
            img = self.transform(img)

        return torch.tensor(
            img, dtype=torch.float32), torch.tensor(
                target, dtype=torch.float32)


def load_model(model='efficientnet-b2'):
    if "efficientnet" in model:
        arch = EfficientNet.from_pretrained(model)
    elif model == "googlenet":
        arch = torchvision.models.googlenet(pretrained=True)
    else:
        arch = torchvision.models.resnet50(pretrained=True)
    model = Net(arch=arch).to(device)
    return model


class Net(nn.Module):
    def __init__(self, arch, return_feats=False):
        super(Net, self).__init__()
        self.arch = arch
        self.return_feats = return_feats
        if 'fgdf' in str(arch.__class__):
            self.arch.fc = nn.Linear(
                in_features=1280,
                out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):
            self.arch._fc = nn.Linear(
                in_features=self.arch._fc.in_features,
                out_features=500, bias=True)
        else:
            self.arch.fc = nn.Linear(
                in_features=arch.fc.in_features,
                out_features=500, bias=True)
        self.ouput = nn.Linear(500, 1)

    def forward(self, images):
        """
        No sigmoid in forward because we are going to
        use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating
        a loss
        """
        x = images
        features = self.arch(x)
        output = self.ouput(features)
        if self.return_feats:
            return features
        return output
