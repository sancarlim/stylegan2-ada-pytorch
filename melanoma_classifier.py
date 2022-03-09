# Last Modified   : 12.01.2022
# By              : Sandra Carrasco <sandra.carrasco@ai.se>

import numpy as np
import pandas as pd
import gc
from argparse import ArgumentParser
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import os

from utils import (seed_everything, confussion_matrix,
                   add_pr_curve_tensorboard,
                   CustomDataset, plot_classes_preds,
                   seed_worker, load_model, testing_transforms,
                   training_transforms, load_synthetic_data,
                   load_isic_data)
import wandb

import warnings
warnings.simplefilter('ignore')

classes = ('benign', 'melanoma')

seed = 2022
seed_everything(seed)

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

writer_path = f'training_classifiers_events/test_all_melanoma/{datetime.datetime.now().month}_{datetime.datetime.now().day}/'
writer = SummaryWriter(writer_path)


# TRAINING
def train(model, train_loader, validate_loader,
          epochs=10, es_patience=3, wandb_flag=False):
    # Training model
    print('Starts training...')

    best_val = 0
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer (gradient descent):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode='max',
        patience=es_patience, verbose=True, factor=0.2)

    loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_auc_history = []
    val_f1_history = []

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
            loss = criterion(output, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

            # Training loss
            running_loss += loss.item()

            # Number of correct training predictions and training accuracy
            train_preds = torch.round(torch.sigmoid(output))

            correct += (
                train_preds.cpu() == labels.cpu().unsqueeze(1)
                ).sum().item()

            if i % 500 == 1:  # == N every N minibatches
                if wandb_flag:
                    wandb.log({'train/training_loss': loss, 'epoch': e})
                else:
                    # Log in Tensorboard
                    writer.add_figure(
                        'predictions vs. actuals',
                        plot_classes_preds(
                            model, images, labels.type(torch.int)),
                        global_step=e+1)

        train_acc = correct / len(training_dataset)
        val_loss, val_auc_score, val_accuracy, val_f1 = val(
            model, validate_loader, criterion)
        training_time = str(
            datetime.timedelta(seconds=time.time() - start_time))[:7]

        print(
          "Epoch: {}/{}.. ".format(e+1, epochs),
          "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
          "Training Accuracy: {:.3f}..".format(train_acc),
          "Validation Loss: {:.3f}.. ".format(val_loss/len(validate_loader)),
          "Validation Accuracy: {:.3f}".format(val_accuracy),
          "Validation AUC Score: {:.3f}".format(val_auc_score),
          "Validation F1 Score: {:.3f}".format(val_f1),
          "Training Time: {}".format(training_time))
        if wandb_flag:
            wandb.log(
                {'train/Training acc': train_acc, 'epoch': e,
                 'val/Validation Acc': val_accuracy,
                 'val/Validation Auc': val_auc_score,
                 'val/Validation Loss': val_loss/len(validate_loader)})
        else:
            # Log in Tensorboard
            writer.add_scalar(
                'training loss', running_loss/len(train_loader), e+1)
            writer.add_scalar(
                'Training acc', train_acc, e+1)
            writer.add_scalar(
                'Validation AUC Score', val_auc_score, e+1)
        scheduler.step(val_accuracy)

        if val_accuracy > best_val:
            best_val = val_accuracy
            if wandb_flag:
                wandb.run.summary["best_auc_score"] = val_auc_score
                wandb.run.summary["best_acc_score"] = val_accuracy
            # Resetting patience since we have new best validation accuracy
            patience = es_patience
            model_path = os.path.join(
                writer_path,
                f'./classifier_{args.model}_{best_val:.4f}'
                f'_{datetime.datetime.now()}.pth')
            # Saving current best model
            torch.save(model.state_dict(), model_path)
            print(f'Saving model in {model_path}')
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping. Best Val f1: {:.3f}'.format(best_val))
                break
        loss_history.append(running_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_auc_history.append(val_auc_score)
        val_f1_history.append(val_f1)

    total_training_time = str(
        datetime.timedelta(seconds=time.time() - Total_start_time))[:7]
    print("Total Training Time: {}".format(total_training_time))
    del train_loader, validate_loader, images
    gc.collect()

    return model_path


def val(model, validate_loader, criterion):
    model.eval()
    preds = []
    all_labels = []
    # Turning off gradients for validation, saves memory and computations
    with torch.no_grad():
        val_loss = 0
        for val_images, val_labels in validate_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_output = model(val_images)
            val_loss += (criterion(val_output, val_labels.view(-1, 1))).item()
            val_pred = torch.sigmoid(val_output)
            preds.append(val_pred.cpu())
            all_labels.append(val_labels.cpu())

        pred = np.vstack(preds).ravel()
        pred2 = torch.tensor(pred)
        val_gt = np.concatenate(all_labels)
        val_gt2 = torch.tensor(val_gt)

        val_accuracy = accuracy_score(val_gt2, torch.round(pred2))
        val_auc_score = roc_auc_score(val_gt, pred)
        val_f1_score = f1_score(val_gt, np.round(pred))

        return val_loss, val_auc_score, val_accuracy, val_f1_score


def test(model, test_loader):
    test_preds = []
    all_labels = []
    with torch.no_grad():

        for _, (test_images, test_labels) in enumerate(test_loader):
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            test_output = model(test_images)
            test_pred = torch.sigmoid(test_output)

            test_preds.append(test_pred.cpu())
            all_labels.append(test_labels.cpu())

        test_pred = np.vstack(test_preds).ravel()
        test_pred2 = torch.tensor(test_pred)
        test_gt = np.concatenate(all_labels)
        test_gt2 = torch.tensor(test_gt)

        '''
        # For edge cases
        indeces_misclassified = np.where(test_gt != np.round(test_pred))[0]
        well_classified = list(
            set(list(range(
                0, len(test_gt2)))) - set(indeces_misclassified.tolist()))
        edge_cases = np.where(
            (test_gt[well_classified] - test_pred[well_classified]
            ) > 0.25)[0]
        '''

        try:
            test_accuracy = accuracy_score(
                test_gt2.cpu(),
                torch.round(test_pred2))
            test_auc_score = roc_auc_score(test_gt, test_pred)
            test_f1_score = f1_score(test_gt, np.round(test_pred))
        except:
            test_auc_score = 0
            test_f1_score = 0
            pass

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_pred2, test_gt2, writer)

    print(
        "Test Accuracy:{:.5f}, ROC_AUC_score: {:.5f}, F1 score: {:.4f}".format(
            test_accuracy, test_auc_score, test_f1_score)
        )

    return test_pred, test_gt, test_accuracy


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--syn_data_path", type=str, default='/workspace/generated-no-valset')
    parser.add_argument(
        "--real_data_path", type=str,
        default='/workspace/melanoma_isic_dataset')
    parser.add_argument(
        "--model", type=str, default='efficientnet-b2',
        choices=["efficientnet-b2", "googlenet", "resnet50"])
    parser.add_argument("--epochs", type=int, default='30')
    parser.add_argument(
        "--es", type=int, default='3', help="Iterations for Early Stopping")
    parser.add_argument(
        "--unbalanced", action='store_true', help='train with 15% melanoma')
    parser.add_argument(
        "--only_reals", action='store_true',
        help='train using only real images')
    parser.add_argument(
        "--only_syn", action='store_true',
        help='train using only synthetic images')
    parser.add_argument("--tags", type=str, default='whole isic')
    parser.add_argument(
        "--synt_n_imgs", type=str, default="0,15",
        help='n benign, n melanoma K synthetic images to add to the real data')
    parser.add_argument(
        "--wandb_flag",
        action="store_true",
        default=False,
        help="Launch experiment and log metrics with wandb",
    )
    args = parser.parse_args()

    if args.wandb_flag:
        wandb.init(project="dai-healthcare", entity='eyeforai', group='isic',
                   tags=[args.tags], config={"model": args.model})
        wandb.config.update(args)

    isic_train_df, validation_df = load_isic_data(args.real_data_path)
    synt_train_df = load_synthetic_data(
        args.syn_data_path, args.synt_n_imgs, args.only_syn)
    if args.only_syn:
        train_df = synt_train_df
    elif args.only_reals:
        train_df = isic_train_df
    else:
        train_df = pd.concat([isic_train_df, synt_train_df])

    training_dataset = CustomDataset(
        df=train_df, train=True, transforms=training_transforms)
    validation_dataset = CustomDataset(
        df=validation_df, train=True,
        transforms=training_transforms)
    testing_dataset = CustomDataset(
        df=validation_df, train=True, transforms=testing_transforms)

    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=32, num_workers=4,
        worker_init_fn=seed_worker, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=16, num_workers=4,
        worker_init_fn=seed_worker, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        testing_dataset, batch_size=16, num_workers=4,
        worker_init_fn=seed_worker, shuffle=False)
    print(len(training_dataset), len(validation_dataset))
    print(len(train_loader), len(validate_loader), len(test_loader))

    # Load model
    model = load_model(args.model)
    print(f'Model {args.model} loaded.')

    # If we need to freeze the pretrained model parameters
    # to avoid backpropogating through them, turn to "False"
    for parameter in model.parameters():
        parameter.requires_grad = True

    # Total Parameters
    # (If the model is unfrozen the trainning params
    # will be the same as the Total params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    model_path = train(model, train_loader,
                       validate_loader,
                       epochs=args.epochs,
                       es_patience=args.es,
                       wandb_flag=args.wandb_flag)

    del training_dataset, validation_dataset
    gc.collect()

    # TESTING THE NETWORK
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    test_pred, test_gt, test_accuracy = test(model, test_loader)

    # CONFUSSION MATRIX
    confussion_matrix(test_gt, test_pred, test_accuracy, writer_path)
