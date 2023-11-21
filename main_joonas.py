from __future__ import print_function

import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import wandb
from monai.transforms import *
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import gc
from model import Attention, GatedAttention, ResNet18Attention, ResNet18AttentionSigmoid

sys.path.append('/gpfs/space/home/joonas97/MIL/AttentionDeepMIL')
from dataloader_TUH import TUH_full_scan_dataset

sys.path.append('/gpfs/space/home/joonas97/MIL')
from utils.utils import find_case_id, attention_accuracy, center_crop_dataframe

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')

CROP_SIZE = 120


def train(model, optimizer, train_loader, loss_function, epoch: int, ROIS: pd.DataFrame, TUH_length: int,
          nth_slice: int, crop_size: int,
          check: bool = False):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    train_loss = 0.
    train_error = 0.
    # attention related accuracies
    train_all_attention_acc = 0.
    train_tumor_only_attention_acc = 0.
    train_all_recall = 0.
    step = 0
    tepoch = tqdm(train_loader, unit="batch", ascii=True)
    for data, bag_label, file_path in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        gc.collect()
        step += 1
        if 'tuh_kidney' in file_path[0]:
            calculate_attention_accuracy = True
            case_id = find_case_id(file_path, start_string='case_', end_string='_0000')

            rois = ROIS.loc[ROIS["file_name"] == "case_" + case_id + "_0000.nii.gz"].copy()[::nth_slice]
            rois = center_crop_dataframe(rois, crop_size)

        else:
            calculate_attention_accuracy = False
        # _, c, x, y, h = data.shape
        # data = torch.reshape(data, (1, h, c, x, y))
        data = torch.permute(data, (0, 4, 1, 2, 3))
        data, bag_label = data.cuda(), bag_label.cuda()

        # reset gradients
        optimizer.zero_grad(set_to_none=True)
        # calculate loss and metrics
        with torch.cuda.amp.autocast():
            Y_prob, Y_hat, A = model.forward(data)
            loss = loss_function(Y_prob, bag_label.float())

        train_loss += loss.item()
        error = model.calculate_classification_error(bag_label, Y_hat)

        train_error += error
        # backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # reset gradients
        optimizer.zero_grad(set_to_none=True)


        if calculate_attention_accuracy:
            attention = A.as_tensor().cpu().detach()[0]
            all_acc, tumor_only_acc, all_recall = attention_accuracy(attention=np.array(attention), df=rois)
            train_all_attention_acc += all_acc
            train_tumor_only_attention_acc += tumor_only_acc
            train_all_recall += all_recall
        if step >= 20 and check:
            break

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    train_all_attention_acc /= TUH_length
    train_tumor_only_attention_acc /= TUH_length
    train_all_recall /= TUH_length
    print('Train loss: {:.4f}, Train error: {:.4f}, attention_accuracy: {:.4f}'.format(train_loss, train_error,
                                                                                       train_all_attention_acc))
    return train_loss, train_error, train_all_attention_acc, train_tumor_only_attention_acc, train_all_recall


def validation(model, test_loader, loss_function, epoch: int, ROIS: pd.DataFrame, TUH_length: int, nth_slice: int,
               crop_size: int,
               check: bool = False):
    # model.eval()
    model.train()

    test_loss = 0.
    test_error = 0.
    step = 0
    # attention related accuracies
    test_all_attention_acc = 0.
    test_tumor_only_attention_acc = 0.
    test_all_recall = 0.
    tepoch = tqdm(test_loader, unit="batch", ascii=True)
    for data, bag_label, file_path in tepoch:
        tepoch.set_description(f"Validation of epoch {epoch}")
        step += 1

        if 'tuh_kidney' in file_path[0]:
            calculate_attention_accuracy = True
            case_id = find_case_id(file_path, start_string='case_', end_string='_0000')
            rois = ROIS.loc[ROIS["file_name"] == "case_" + case_id + "_0000.nii.gz"].copy()[::nth_slice]
            rois = center_crop_dataframe(rois, crop_size)
        else:
            calculate_attention_accuracy = False

        # _, c, x, y, h = data.shape # should permute to --> (1, h, c, x, y)
        data = torch.permute(data, (0, 4, 1, 2, 3))
        data, bag_label = data.cuda(), bag_label.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            Y_prob, predicted_label, attention_weights = model.forward(data)

            loss = loss_function(Y_prob, bag_label.float())
            test_loss += loss.item()  # .data[0]
            error = model.calculate_classification_error(bag_label, predicted_label)
            test_error += error
        if calculate_attention_accuracy:
            attention = attention_weights.as_tensor().cpu().detach()[0]
            all_acc, tumor_only_acc, all_recall = attention_accuracy(attention=np.array(attention), df=rois)
            test_all_attention_acc += all_acc
            test_tumor_only_attention_acc += tumor_only_acc
            test_all_recall += all_recall
        if step >= 20 and check:
            break

    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    test_all_attention_acc /= TUH_length
    test_tumor_only_attention_acc /= TUH_length
    test_all_recall /= TUH_length
    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}, attention_accuracy: {:.4f}'.format(test_loss, test_error,
                                                                                            test_all_attention_acc))
    return test_loss, test_error, test_all_attention_acc, test_tumor_only_attention_acc, test_all_recall


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, "config.yaml")
    print(f"Running {cfg.project}, Work in {os.getcwd()}")

    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    transforms_train = Compose(
        [
            RandRotate(range_x=1, prob=1),
            RandGaussianNoise(prob=0.5, mean=0, std=0.2),
            RandAffine(prob=0.5, scale_range=(-0.1, 0.1), translate_range=(-50, 50),
                       padding_mode="border")
        ])

    train_dataset = TUH_full_scan_dataset(datasets=cfg.data.datasets, dataset_type="train",
                                          only_every_nth_slice=cfg.data.take_every_nth_slice,
                                          downsample=cfg.data.downsample, as_rgb=True,
                                          sample_shifting=cfg.data.sample_shifting,
                                          augmentations=transforms_train if cfg.data.data_augmentations is True else None,
                                          plane=cfg.data.plane, center_crop=cfg.data.crop_size,
                                          artificial_spheres=cfg.data.artifical_spheres)
    test_dataset = TUH_full_scan_dataset(datasets=cfg.data.datasets, dataset_type="test",
                                         only_every_nth_slice=cfg.data.take_every_nth_slice,
                                         downsample=cfg.data.downsample, as_rgb=True, plane=cfg.data.plane,
                                         center_crop=cfg.data.crop_size,
                                         artificial_spheres=cfg.data.artifical_spheres)
    train_loader = data_utils.DataLoader(train_dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         **loader_kwargs)

    test_loader = data_utils.DataLoader(test_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)

    TUH_length_train = train_dataset.TUH_length
    TUH_length_test = test_dataset.TUH_length

    # read in statistics about the scans for validation metrics
    base_path = '/gpfs/space/home/joonas97/MIL/AttentionDeepMIL/'
    test_ROIS = pd.read_csv(base_path + "ROIS/{0}_test_ROIS.csv".format(cfg.data.plane))
    train_ROIS = pd.read_csv(base_path + "ROIS/{0}_train_ROIS.csv".format(cfg.data.plane))

    logging.info('Init Model')

    if cfg.model.name == 'resnet18':
        model = ResNet18Attention(
            neighbour_range=cfg.model.neighbour_range)

        if cfg.model.freeze_backbone:
            # Let's freeze the backbone
            model.backbone.requires_grad_(False)

    elif cfg.model.name == 'resnet18sigmoid':
        model = ResNet18AttentionSigmoid(
            neighbour_range=cfg.model.neighbour_range)

    elif cfg.model.name == 'attention':
        model = Attention()
    elif cfg.model.name == 'gated_attention':
        model = GatedAttention()

        # if you need to continue training
    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        model.load_state_dict(
            torch.load(os.path.join('/gpfs/space/home/joonas97/MIL/AttentionDeepMIL/', cfg.checkpoint)))
    if torch.cuda.is_available():
        model.cuda()

    # summary(model, (216, 216, 216))
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.999),
                           weight_decay=cfg.training.weight_decay)
    loss_function = torch.nn.BCEWithLogitsLoss().cuda()

    if not cfg.check:
        experiment = wandb.init(project='MIL experiment on TUH kidney study', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.training.epochs,
                 learning_rate=cfg.training.learning_rate, model_name=cfg.model.name,
                 weight_decay=cfg.training.weight_decay, directory=os.getcwd()))
    logging.info('Start Training')

    best_test_error = 1  # should be 1
    best_test_loss = 10
    best_epoch = 0
    best_attention = 0
    not_improved_epochs = 0

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_error, train_att_all, train_att_tumor, train_att_recall = train(model, optimizer,
                                                                                          train_loader,
                                                                                          loss_function, epoch,
                                                                                          ROIS=train_ROIS,
                                                                                          TUH_length=TUH_length_train,
                                                                                          nth_slice=cfg.data.take_every_nth_slice,
                                                                                          crop_size=cfg.data.crop_size,
                                                                                          check=cfg.check)
        test_loss, test_error, test_att_all, test_att_tumor, test_att_recall = validation(model, test_loader,
                                                                                          loss_function, epoch,
                                                                                          ROIS=test_ROIS,
                                                                                          TUH_length=TUH_length_test,
                                                                                          nth_slice=cfg.data.take_every_nth_slice,
                                                                                          crop_size=cfg.data.crop_size,
                                                                                          check=cfg.check)
        if cfg.check:
            logging.info("Model check completed")
            return
        if test_error < best_test_error:

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            logging.info(f"Best new model at epoch {epoch} (smallest test error)!")

            best_test_error = test_error
            best_epoch = epoch
            not_improved_epochs = 0
        elif test_att_all > best_attention:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_attention_model.pth'))
            logging.info(f"Best new attention at epoch {epoch}!")

            best_attention = test_att_all
            best_epoch = epoch
            not_improved_epochs = 0
        else:
            if not_improved_epochs > 20:
                logging.info("Model has not improved for the last 20 epochs, stopping training")
                break
            not_improved_epochs += 1
        experiment.log({
            'train loss': train_loss,
            'test loss': test_loss,
            'train error': train_error,
            'test error': test_error,
            'train_attention_accuracy': train_att_all,
            'test_attention_accuracy': test_att_all,
            'train_tumor_attention_accuracy': train_att_tumor,
            'test_tumor_attention_accuracy': test_att_tumor,
            'train_attention_recall': train_att_recall,
            'test_attention_recall': test_att_recall,
            'epoch': epoch})

    torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
    logging.info(f'Last checkpoint! Checkpoint {epoch} saved!')
    logging.info(f"Training completed, best_metric: {best_test_error:.4f} at epoch: {best_epoch}")


if __name__ == "__main__":
    main()
