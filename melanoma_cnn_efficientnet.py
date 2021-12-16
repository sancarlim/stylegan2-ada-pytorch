import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import gc
import cv2
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models, utils
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset
# from imblearn.under_sampling import RandomUnderSampler

import time
from datetime import datetime
import random

from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

from efficientnet_pytorch import EfficientNet

import os 

import warnings
warnings.simplefilter('ignore')

classes = ('benign', 'melanoma')
# Creating seeds to make results reproducible
def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 1234
seed_everything(seed)

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

writer = SummaryWriter('training_classifiers_events/')

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


def confussion_matrix(test, test_pred, test_accuracy):
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

    now=datetime.now()
    plt.savefig(f'./conf_matrix_{test_accuracy:.4f}_{now.strftime("%d_%m_%H:%M")}.png')
    writer.add_image('conf_matrix', fig)


def plot_diagnosis(model, predict_image_path):
    img_nb = predict_image_path.split('/')[-1].split('.')[0]
    probs, classes = predict(predict_image_path, model)   
    print(probs)
    print(classes)

    # Display an image along with the diagnosis of melanoma or benign
    # Plot Skin image input image
    plt.figure(figsize = (6,10))
    plot_1 = plt.subplot(2,1,1)

    image = process_image(predict_image_path)

    imshow(image, plot_1)
    font = {"color": 'g'} if 'Benign' in classes else {"color": 'r'}
    plot_1.set_title(f"Diagnosis: {classes}, Output (prob) {probs[0]:.4f}", fontdict=font);
    plt.savefig(f'./prediction_{img_nb}.png')


def create_split(source_dir, n_b, n_m):       
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if os.path.isfile(f)]
    
    ind_0, ind_1 = [], []
    for i, f in enumerate(input_images):
        if f.split('.jpg')[0][-1] == '0' or f.split('.png')[0][-1] == '0':
            ind_0.append(i)
        else:
            ind_1.append(i) 

    ind_0=np.random.permutation(ind_0[::2])[:n_b*1000]
    ind_1=np.random.permutation(ind_1[1::2])[:n_m*1000]

    # ind_1 = ind_1[:round(len(ind_1)*0.16)] #Train with 15% melanoma
    
    train_id_list, val_id_list  = ind_0[:round(len(ind_0)*0.8)],  ind_0[round(len(ind_0)*0.8):]       #ind_0[round(len(ind_0)*0.6):round(len(ind_0)*0.8)] ,
    train_id_1, val_id_1 = ind_1[:round(len(ind_1)*0.8)],  ind_1[round(len(ind_1)*0.8):] #ind_1[round(len(ind_1)*0.6):round(len(ind_1)*0.8)] ,
    
    train_id_list = np.append(train_id_list, train_id_1)
    val_id_list =   np.append(val_id_list, val_id_1)
    #test_id_list = np.append(test_id_list, test_id_1)     
    
    return train_id_list, val_id_list  #test_id_list

# helper functions

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


def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
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
        #images = Image.open(img_path)
        images =Image.open(img_path)

        if self.transforms:
            images = self.transforms(images)

        if self.train:
            labels = self.df.iloc[index]['target']
            #return images, labels
            return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
        
        else:
            #return (images)
            return torch.tensor(images, dtype=torch.float32)
    

class Synth_Dataset(Dataset):
    def __init__(self, source_dir, transform, id_list=None, input_img=None, test = False, unbalanced=False):
        self.transform = transform
        self.source_dir = source_dir
        
        if input_img is None:
            self.input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if os.path.isfile(f)]
        else:
            self.input_images = input_img
        
        if unbalanced:
            ind_0, ind_1 = create_split(args.syn_data_path, unbalanced=unbalanced)
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


class Net(nn.Module):
    def __init__(self, arch):
        super(Net, self).__init__()
        self.arch = arch
        if 'fgdf' in str(arch.__class__):
            self.arch.fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):   
            self.arch._fc = nn.Linear(in_features=1408, out_features=500, bias=True)
            #self.dropout1 = nn.Dropout(0.2)
            
        self.ouput = nn.Linear(500, 1)
        
    def forward(self, images):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x = images
        features = self.arch(x)
        output = self.ouput(features)
        
        return output


### TRAINING ###
def train(model, train_loader, validate_loader, validate_loader_reals, k_fold = 0, epochs = 10, es_patience = 3):
    # Training model
    print('Starts training...')

    best_val = 0
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer (gradient descent):
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)

    loss_history=[]  
    train_acc_history=[]  
    val_loss_history=[]  
    val_acc_history=[] 
    val_auc_history=[]
    val_f1_history=[]

    val_loss_r_history=[] 
    val_auc_r_history=[]
        
    patience = es_patience
    Total_start_time = time.time()  
    model.to(device)

    for e in range(epochs):
        
        start_time = time.time()
        correct = 0
        running_loss = 0
        model.train()
        
        for i, (images, labels) in enumerate(train_loader):
            
            images, labels = images.to(device), labels.to(device)
                
            optimizer.zero_grad()
            
            output = model(images) 
            loss = criterion(output, labels.view(-1,1))  
            loss.backward()
            optimizer.step()
            
            # Training loss
            running_loss += loss.item()

            # Number of correct training predictions and training accuracy
            train_preds = torch.round(torch.sigmoid(output))
                
            correct += (train_preds.cpu() == labels.cpu().unsqueeze(1)).sum().item()
            
            if i % 500 == 1:  # == N every N minibatches 
                writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(model, images, labels.type(torch.int)),
                            global_step=e+1)
                            
        train_acc = correct / len(training_dataset)

        val_loss, val_auc_score, val_accuracy, val_f1 = val(model, validate_loader, criterion)
        #val_loss_r, val_auc_score_r, val_accuracy_r, val_f1_r = val(model, validate_loader_reals, criterion)
        

        training_time = str(datetime.timedelta(seconds=time.time() - start_time))[:7]
            
        print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
            "Training Accuracy: {:.3f}..".format(train_acc),
            "Validation Loss: {:.3f}.. ".format(val_loss/len(validate_loader)),
            "Validation Accuracy: {:.3f}".format(val_accuracy),
            "Validation AUC Score: {:.3f}".format(val_auc_score),
            "Validation F1 Score: {:.3f}".format(val_f1),
            "Training Time: {}".format( training_time))
            
        writer.add_scalar('training loss',  running_loss/len(train_loader), e+1 )
        writer.add_scalar('Training acc', train_acc, e+1 )
        writer.add_scalar('Validation AUC Score', val_auc_score, e+1 )
        

        scheduler.step(val_auc_score)
                
        if val_auc_score > best_val:
            best_val = val_auc_score
            patience = es_patience  # Resetting patience since we have new best validation accuracy
            bal_unbal = args.syn_data_path.split('/')[-1]
            model_path = f'./melanoma_model_{k_fold}_{best_val:.4f}_{bal_unbal}.pth'
            torch.save(model.state_dict(), model_path)  # Saving current best model
            print(f'Saving model in {model_path}')
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping. Best Val f1: {:.3f}'.format(best_val))
                break
        
        loss_history.append(running_loss)  
        train_acc_history.append(train_acc)    
        val_loss_history.append(val_loss)  
        #val_acc_history.append(val_accuracy)
        val_auc_history.append(val_auc_score)
        val_f1_history.append(val_f1)

        #val_loss_r_history.append(val_loss_r)  
        #val_auc_r_history.append(val_auc_score_r)

    total_training_time = str(datetime.timedelta(seconds=time.time() - Total_start_time  ))[:7]                  
    print("Total Training Time: {}".format(total_training_time))

    del train_loader, validate_loader, images
    gc.collect()

    return loss_history, train_acc_history, val_auc_history, val_loss_history, val_f1_history, model_path #, val_loss_r_history, val_auc_r_history
                
def val(model, validate_loader, criterion):          
    model.eval()
    preds=[]            
    all_labels=[]
    # Turning off gradients for validation, saves memory and computations
    with torch.no_grad():
        
        val_loss = 0
        val_correct = 0
    
        for val_images, val_labels in validate_loader:
        
            val_images, val_labels = val_images.to(device), val_labels.to(device)
        
            val_output = model(val_images)
            val_loss += (criterion(val_output, val_labels.view(-1,1))).item() 
            val_pred = torch.sigmoid(val_output)
            
            preds.append(val_pred.cpu())
            all_labels.append(val_labels.cpu())
        pred=np.vstack(preds).ravel()
        pred2 = torch.tensor(pred)
        val_gt = np.concatenate(all_labels)
        val_gt2 = torch.tensor(val_gt)
            
        val_accuracy = accuracy_score(val_gt2, torch.round(pred2))
        val_auc_score = roc_auc_score(val_gt, pred)
        val_f1_score = f1_score(val_gt, np.round(pred))

        return val_loss, val_auc_score, val_accuracy, val_f1_score

def test(model, test_loader):
    test_preds=[]
    all_labels=[]
    with torch.no_grad():
        
        for _, (test_images, test_labels) in enumerate(test_loader):
            
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            
            test_output = model(test_images)
            test_pred = torch.sigmoid(test_output)
                
            test_preds.append(test_pred.cpu())
            all_labels.append(test_labels.cpu())
            
        test_pred=np.vstack(test_preds).ravel()
        test_pred2 = torch.tensor(test_pred)
        test_gt = np.concatenate(all_labels)
        test_gt2 = torch.tensor(test_gt)
        try:
            test_accuracy = accuracy_score(test_gt2.cpu(), torch.round(test_pred2))
            test_auc_score = roc_auc_score(test_gt, test_pred)
            test_f1_score = f1_score(test_gt, np.round(test_pred))
        except:
            test_auc_score = 0
            test_f1_score = 0
            pass
    
    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_pred2, test_gt2)

    print("Test Accuracy: {:.5f}, ROC_AUC_score: {:.5f}, F1 score: {:.4f}".format(test_accuracy, test_auc_score, test_f1_score))  

    return test_pred, test_gt, test_accuracy


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--syn_data_path", type=str, default='/workspace/generated-aug-bg/')
    parser.add_argument("--real_data_path", type=str, default='/workspace/melanoma_isic_dataset')
    parser.add_argument("--epochs", type=int, default='15')
    parser.add_argument("--kfold", type=int, default='3', help='number of folds for stratisfied kfold')
    parser.add_argument("--unbalanced", action='store_true', help='train with 15% melanoma')
    parser.add_argument("--only_syn", action='store_true', help='train using only synthetic images')
    parser.add_argument("--synt_n_imgs",  type=str, default="0,15", help='n benign, n melanoma K synthetic images to add to the real data')
    args = parser.parse_args()

    # For training with ISIC dataset

    df = pd.read_csv(os.path.join(args.real_data_path , 'train_concat.csv'))
    # test_df = pd.read_csv(os.path.join(args.data_path ,'melanoma_external_256/test.csv'))
    # test_img_dir = os.path.join(args.data_path , 'melanoma_external_256/test/test/')
    train_img_dir = os.path.join(args.real_data_path ,'train/train/')
    
    df['image_name'] = [os.path.join(train_img_dir, df.iloc[index]['image_name'] + '.jpg') for index in range(len(df))]

    train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42) 
    train_df=pd.DataFrame(train_split)
    validation_df=pd.DataFrame(valid_split)

    # under_sampler = RandomUnderSampler(random_state=42)
    # train_df_res, _ = under_sampler.fit_resample(train_df, train_df.target)
    
    # Defining transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([#Microscope(),
                                            #AdvancedHairAugmentation(),
                                            transforms.RandomRotation(30),
                                            #transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            #transforms.ColorJitter(brightness=32. / 255.,saturation=0.5,hue=0.01),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(256),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(256),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    
    """
    train_0, test_0 = train_test_split(ind_0, test_size=0.2, random_state=3)
    train_1, test_1 = train_test_split(ind_1, test_size=0.2, random_state=3)
    train_id = np.append(train_0,train_1)
    test_ix = np.append(test_0,test_1)
    train_0, val_0 = train_test_split(ind_0, test_size=0.25, random_state=3) # 0.25 x 0.8 = 0.2
    train_1, val_1 = train_test_split(ind_1, test_size=0.25, random_state=3) # 0.25 x 0.8 = 0.2
    train_id = np.append(train_0,train_1)
    val_id = np.append(val_0, val_1)
    """
    input_images = [str(f) for f in sorted(Path(args.syn_data_path).rglob('*')) if os.path.isfile(f)]
    y = [0 if f.split('.jpg')[0][-1] == '0' else 1 for f in input_images]
    
    n_b, n_m = [int(i) for i in args.synt_n_imgs.split(',') ] if not args.only_syn else [1000,1000]
    train_id_list, val_id_list = create_split(args.syn_data_path, n_b , n_m)
    # ind=np.append(ind_0, ind_1)
    train_img = [input_images[int(i)] for i in train_id_list]
    train_gt = [y[int(i)] for i in train_id_list]
    # train_img, test_img, train_gt, test_gt = train_test_split(input_images, y, stratify=y, test_size=0.2, random_state=3)
    synt_train_df = pd.DataFrame({'image_name': train_img, 'target': train_gt})
    if args.only_syn:
        train_df = synt_train_df
    else:
        train_df = pd.concat([train_df, synt_train_df]) 
    
    fold=0
    #skf = StratifiedKFold(n_splits=args.kfold)
    #for fold, (train_ix, val_ix) in enumerate(skf.split(train_img, train_gt)): 
    #    print(len(train_ix), len(val_ix))                                      
    #    #train_df = df.iloc[train_ix].reset_index(drop=True)
    #    #validation_df = df.iloc[val_ix].reset_index(drop=True)

    #    # Loading the datasets with the transforms previously defined
    #    #train_id, val_id, test_id = create_split(args.data_path, unbalanced=args.unbalanced)
                       
                                   
    # training_dataset = Synth_Dataset(source_dir = args.syn_data_path, transform = training_transforms, id_list = None, unbalanced=args.unbalanced)  # CustomDataset(df = train_df_res, img_dir = train_img_dir,  train = True, transforms = training_transforms )
    # train_id, val_id = create_split(args.syn_data_path, unbalanced=args.unbalanced)
    training_dataset = CustomDataset(df = train_df, train = True, transforms = training_transforms )
                        #Synth_Dataset(source_dir = args.syn_data_path, transform = training_transforms, id_list = train_id, unbalanced=args.unbalanced)  # CustomDataset(df = train_df_res, img_dir = train_img_dir,  train = True, transforms = training_transforms )
    validation_dataset =  CustomDataset(df = validation_df, train = True, transforms = training_transforms) 

    #testing_dataset = Synth_Dataset(source_dir = args.data_path, transform = testing_transforms, id_list = range(len(test_gt)), input_img=test_img)
    testing_dataset = CustomDataset(df = validation_df, train = True, transforms = testing_transforms ) 
                   

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=32, num_workers=4, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle = False)
    validate_loader_real = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle = False)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, shuffle = False)
    print(len(training_dataset), len(validation_dataset))
    print(len(train_loader),len(validate_loader),len(test_loader))

    dataiter = iter(train_loader)
    imgs, labels =dataiter.next()
    imgs_list = [renormalize(img) for img in imgs]
    img_grid = utils.make_grid(imgs_list)
    writer.add_image('train_loader_images', img_grid)

    arch = EfficientNet.from_pretrained('efficientnet-b2')
    model = Net(arch=arch)  
    model = model.to(device)
    
    writer.add_graph(model, imgs.to(device))

    # If we need to freeze the pretrained model parameters to avoid backpropogating through them, turn to "False"
    for parameter in model.parameters():
        parameter.requires_grad = True

    #Total Parameters (If the model is unfrozen the trainning params will be the same as the Total params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    loss_history, train_acc_history, val_auc_history, val_loss_history, val_f1_history, model_path = train(model, train_loader, validate_loader, validate_loader_real, fold, epochs=args.epochs, es_patience=3)

    del training_dataset, validation_dataset 
    gc.collect()

    
    ### TESTING THE NETWORK ###
    #model_path = '/home/stylegan2-ada-pytorch/models_trained_with_synth/melanoma_model_unbal_0.9999899287601407.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    test_pred, test_gt, test_accuracy = test(model, test_loader)  

    ### CONFUSSION MATRIX ###
    confussion_matrix(test_gt, test_pred, test_accuracy)
    