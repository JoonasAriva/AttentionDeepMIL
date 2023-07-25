from __future__ import print_function

import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import wandb
from monai.transforms import *
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from model import Attention, GatedAttention, ResNet18Attention

sys.path.append('/gpfs/space/home/joonas97/MIL/AttentionDeepMIL')
from dataloader_TUH import TUH_full_scan_dataset

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')


def train(model, optimizer, train_loader, epoch, loss_function, check=False):
    model.train()
    train_loss = 0.
    train_error = 0.
    step = 0
    tepoch = tqdm(train_loader, unit="batch", ascii=True)
    for data, bag_label in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        step += 1

        #_, c, x, y, h = data.shape
        #data = torch.reshape(data, (1, h, c, x, y))
        data = torch.permute(data, (0, 4, 1, 2, 3))
        data, bag_label = data.cuda(), bag_label.cuda()

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        with torch.cuda.amp.autocast():
            Y_prob, Y_hat, A = model.forward(data)
            loss = loss_function(Y_prob, bag_label.float())
            train_loss += loss.item()
            error = model.calculate_classification_error(bag_label, Y_hat)

        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

        if step >= 20 and check:
            break

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    print('Train loss: {:.4f}, Train error: {:.4f}'.format(train_loss, train_error))
    return train_loss, train_error


def validation(model, test_loader, loss_function, epoch, val_dict, check=False):
    # model.eval()
    model.train()

    test_loss = 0.
    test_error = 0.
    step = 0
    tepoch = tqdm(test_loader, unit="batch", ascii=True)
    for data, bag_label in tepoch:
        tepoch.set_description(f"Validation of epoch {epoch}")
        step += 1

        #_, c, x, y, h = data.shape # should permute to --> (1, h, c, x, y)
        data = torch.permute(data, (0, 4, 1, 2, 3))
        data, bag_label = data.cuda(), bag_label.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            Y_prob, predicted_label, attention_weights = model.forward(data)

            loss = loss_function(Y_prob, bag_label.float())
            test_loss += loss.item()  # .data[0]
            error = model.calculate_classification_error(bag_label, predicted_label)
            test_error += error

        if str(step) in val_dict:
            val_dict[str(step)][0].append(int(predicted_label.as_tensor().cpu().item()))
        else:
            val_dict[str(step)] = ([], [data.shape])
            val_dict[str(step)][0].append(int(predicted_label.as_tensor().cpu().item()))
        # print(val_dict)
        if step >= 20 and check:
            break

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))
    return test_loss, test_error, val_dict


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Running {cfg.project}, Work in {os.getcwd()}")

    np.random.seed(cfg.training.seed)

    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    transforms_train = Compose(
        [
            RandRotate(range_x=1, prob=1),
            RandGaussianNoise(prob=0.5, mean=0, std=0.2),
            RandAffine(prob=0.5, scale_range=(-0.1, 0.1), translate_range=(-50, 50),
                       padding_mode="border")
        ])

    train_dataset = TUH_full_scan_dataset(dataset_type="train", only_every_nth_slice=cfg.data.take_every_nth_slice,
                                          interpolation=cfg.data.interpolation, as_rgb=True,
                                          sample_shifting=cfg.data.sample_shifting,
                                          augmentations=transforms_train if cfg.data.data_augmentations is True else None,
                                          plane = cfg.data.plane)
    test_dataset = TUH_full_scan_dataset(dataset_type="test", only_every_nth_slice=cfg.data.take_every_nth_slice,
                                         interpolation=cfg.data.interpolation, as_rgb=True, plane = cfg.data.plane)
    train_loader = data_utils.DataLoader(train_dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         **loader_kwargs)

    test_loader = data_utils.DataLoader(test_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)

    logging.info('Init Model')

    if cfg.model.name == 'resnet18':
        model = ResNet18Attention(
            neighbour_range=cfg.model.neighbour_range)
        # Let's freeze the backbone
        # model.backbone.requires_grad_(False)
    elif cfg.model.name == 'attention':
        model = Attention()
    elif cfg.model.name == 'gated_attention':
        model = GatedAttention()

        # if you need to continue training
    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        model.load_state_dict(torch.load(os.path.join('/gpfs/space/home/joonas97/MIL/AttentionDeepMIL/', cfg.checkpoint)))
    if torch.cuda.is_available():
        model.cuda()


    # summary(model, (3, 512, 512))
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.999),
                           weight_decay=cfg.training.weight_decay)
    loss_function = torch.nn.BCEWithLogitsLoss().cuda()
    if not cfg.check:
        experiment = wandb.init(project='MIL experiment on TUH kidney study', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.training.epochs,
                 learning_rate=cfg.training.learning_rate, model_name=cfg.model.name,
                 weight_decay=cfg.training.weight_decay))
    logging.info('Start Training')

    best_test_error = 1  # should be 1
    best_test_loss = 10
    best_epoch = 0
    not_improved_epochs = 0

    val_dict = dict()
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_error = train(model, optimizer, train_loader, epoch, loss_function, check=cfg.check)
        test_loss, test_error, val_dict = validation(model, test_loader, loss_function, epoch, val_dict,
                                                     check=cfg.check)
        # for k, v in val_dict.items():
        #    print(k, v)
        if cfg.check:
            logging.info("Model check completed")
            return
        if test_loss < best_test_loss:

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            logging.info(f"Best new model at epoch {epoch}!")
            # best_test_error = test_error
            best_test_loss = test_loss
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
            'epoch': epoch})

    torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
    logging.info(f'Last checkpoint! Checkpoint {epoch} saved!')
    logging.info(f"Training completed, best_metric: {best_test_error:.4f} at epoch: {best_epoch}")


if __name__ == "__main__":
    main()
