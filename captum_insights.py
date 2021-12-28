import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature
from efficientnet_pytorch import EfficientNet
from melanoma_cnn_efficientnet import Net, CustomDataset


# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testing_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
def get_classes():
    classes = [
        "Non-melanoma",
        "Melanoma", 
    ]
    return classes

def get_pretrained_model():
    arch = EfficientNet.from_pretrained('efficientnet-b2')
    model = Net(arch=arch)  
    # summary(model, (3, 256, 256), device='cpu')
    model.load_state_dict(torch.load('/workspace/stylegan2-ada-pytorch/CNN_trainings/melanoma_model_0_0.9225_16_12_train_reals+15melanoma.pth'))
    model.to(device)
    return model

def baseline_func(input):
    return input * 0 # +256


def formatted_data_iter():
    # ISIC dataset
    df = pd.read_csv('/workspace/melanoma_isic_dataset/train_concat.csv') 
    train_img_dir = os.path.join('/workspace/melanoma_isic_dataset/train/train/')
    
    train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42) 
    validation_df=pd.DataFrame(valid_split)
    validation_df['image_name'] = [os.path.join(train_img_dir, validation_df.iloc[index]['image_name'] + '.jpg') for index in range(len(validation_df))]
    testing_dataset = CustomDataset(df = test_df, train = True, transforms = testing_transforms ) 
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, shuffle = False)         
    while True:
        images, labels = next(dataloader)
        yield Batch(inputs=images, labels=labels)


# Run the visualizer and render inside notebook for interactive debugging

model = get_pretrained_model()
visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: torch.nn.functional.softmax(o, 1),
    classes=get_classes(),
    features=[
        ImageFeature(
            "Photo",
            baseline_transforms=[baseline_func],
            input_transforms=[testing_transforms],
        )
    ],
    dataset=formatted_data_iter(),
)
visualizer.render()
