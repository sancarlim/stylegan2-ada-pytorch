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
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

from pathlib import Path
import random

from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

classes = ('benign', 'melanoma')

# Defining transforms for the training, validation, and testing sets
training_transforms = torchvision.transforms.Compose([#Microscope(),
                                        #AdvancedHairAugmentation(),
                                        torchvision.transforms.RandomRotation(30),
                                        #transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                                        torchvision.transforms.RandomHorizontalFlip(),
                                        torchvision.transforms.RandomVerticalFlip(),
                                        #transforms.ColorJitter(brightness=32. / 255.,saturation=0.5,hue=0.01),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

validation_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                            torchvision.transforms.CenterCrop(256),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

testing_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
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
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
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
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
  
    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
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
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=1): #just 2 classes from 1 single output
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''   
    image = process_image(image_path)
    
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    #print(image.shape)
    #print(type(image))
    
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    
    output = model(image)
    
    probabilities = torch.sigmoid(output)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    top_classes = []
    
    if probabilities > 0.5 :
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
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def renormalize(tensor):
    minFrom= tensor.min()
    maxFrom= tensor.max()
    minTo = 0
    maxTo=1
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
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def add_pr_curve_tensorboard(class_index, test_probs, test_label, writer, global_step=0):
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
                        index = ['Benign','Malignant'], 
                        columns = ['Benign','Malignant'])

    fig = plt.figure(figsize=(5.5,4))
    sb.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix \nAccuracy:{0:.3f}'.format(test_accuracy))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    now=datetime.datetime.now()
    # writer.add_image('conf_matrix', fig)
    plt.savefig(os.path.join(writer_path, f'conf_matrix_{test_accuracy:.4f}_{now.strftime("%d_%m_%H_%M")}.png'))


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
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    images = transformer.forward(images).to('cuda')
    return images.requires_grad_(True)

def load_isic_data(path):
    # ISIC dataset 
    df = pd.read_csv(os.path.join(path , 'train_concat.csv'))
    # test_df = pd.read_csv(os.path.join(args.data_path ,'melanoma_external_256/test.csv'))
    # test_img_dir = os.path.join(args.data_path , 'melanoma_external_256/test/test/')
    train_img_dir = os.path.join(path ,'train/train/')
    
    df['image_name'] = [os.path.join(train_img_dir, df.iloc[index]['image_name'] + '.jpg') for index in range(len(df))]

    train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42) 
    train_df=pd.DataFrame(train_split)
    validation_df=pd.DataFrame(valid_split)
    return train_df, validation_df

def load_synthetic_data(syn_data_path, synt_n_imgs, only_syn=False):
    #Load all images and labels from path
    input_images = [str(f) for f in sorted(Path(syn_data_path).rglob('*')) if os.path.isfile(f)]
    y = [0 if f.split('.jpg')[0][-1] == '0' else 1 for f in input_images]
    
    ind_0, ind_1 = [], []
    for i, f in enumerate(input_images):
        if f.split('.')[0][-1] == '0':
            ind_0.append(i)
        else:
            ind_1.append(i) 

    # Select number of melanomas and benign samples
    n_b, n_m = [int(i) for i in synt_n_imgs.split(',') ] if not only_syn else [1000,1000]
    ind_0=np.random.permutation(ind_0)[:n_b*1000]
    ind_1=np.random.permutation(ind_1)[:n_m*1000]

    id_list = np.append(ind_0, ind_1) 

    train_img = [input_images[int(i)] for i in id_list]
    train_gt = [y[int(i)] for i in id_list]
    # train_img, test_img, train_gt, test_gt = train_test_split(input_images, y, stratify=y, test_size=0.2, random_state=3)
    train_df = pd.DataFrame({'image_name': train_img, 'target': train_gt})
    
    return train_df 


class AdvancedHairAugmentation:
    """
    Impose an image of a hair to the target image

    Args:
        hairs (int): maximum number of hairs to impose
        hairs_folder (str): path to the folder with hairs images
    """

    def __init__(self, hairs: int = 5, hairs_folder: str = "../input/melanoma-hairs"):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        n_hairs = random.randint(0, self.hairs)
        
        if not n_hairs:
            return img
        
        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]
        
        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            # Creating a mask and inverse mask
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of hair in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of hair from hair image.
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            # Put hair in ROI and modify the target image
            dst = cv2.add(img_bg, hair_fg)

            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
                
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, hairs_folder="{self.hairs_folder}")'

class DrawHair:
    """
    Draw a random number of pseudo hairs

    Args:
        hairs (int): maximum number of hairs to draw
        width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs:int = 4, width:tuple = (1, 2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        if not self.hairs:
            return img
        
        width, height, _ = img.shape
        
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line 
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(img, origin, end, color, random.randint(self.width[0], self.width[1]))
        
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'

class Microscope:
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Args:
        p (float): probability of applying an augmentation
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8), # image placeholder
                        (img.shape[0]//2, img.shape[1]//2), # center point of circle
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15), # radius
                        (0, 0, 0), # color
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)
        
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True, transforms= None):
        self.df = df
        self.transforms = transforms
        self.train = train
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        img_path = self.df.iloc[index]['image_name']
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        images =Image.open(img_path)

        if self.transforms:
            images = self.transforms(images)
            
        labels = self.df.iloc[index]['target']

        if self.train:
            #return images, labels
            return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
        
        else:
            #return (images)
            return img_path, torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
    

class Synth_Dataset(Dataset):
    def __init__(self, source_dir, transform, id_list=None, input_img=None, test = False, unbalanced=False):
        self.transform = transform
        self.source_dir = source_dir
        
        if input_img is None:
            self.input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if os.path.isfile(f)]
        else:
            self.input_images = input_img
        
        if unbalanced:
            ind_0, ind_1 = create_split(source_dir, unbalanced=unbalanced)
            ind=np.append(ind_0, ind_1)
            self.input_images = [self.input_images[i] for i in ind]

        self.id_list = id_list if id_list is not None else range(len(self.input_images))
            
        if test:
            if unbalanced:
                self.input_images = self.input_images[:5954]
        
    def __len__(self):
        return len(self.id_list)
        
    def __getitem__(self, idx): 
        idx = self.id_list[idx]

        image_fn = self.input_images[idx]   #f'{idx:04d}_{idx%2}'

        img = np.array(Image.open(image_fn))
        target = int(image_fn.split('_')[-1].replace('.jpg',''))  
        
        if self.transform is not None:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)



def load_model(model = 'efficientnet-b2'):
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
            self.arch.fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):   
            self.arch._fc = nn.Linear(in_features=self.arch._fc.in_features, out_features=500, bias=True)
            #self.dropout1 = nn.Dropout(0.2)
        else:   
            self.arch.fc = nn.Linear(in_features=arch.fc.in_features, out_features=500, bias=True)
            
        self.ouput = nn.Linear(500, 1)
        
    def forward(self, images):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x = images
        features = self.arch(x)
        output = self.ouput(features)
        if self.return_feats:
            return features
        return output