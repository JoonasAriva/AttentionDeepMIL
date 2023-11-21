from __future__ import print_function

import logging
import os
import sys
from pathlib import Path

import hydra
import monai
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import wandb
from monai.networks.nets import resnet18
from monai.transforms import *
from omegaconf import OmegaConf, DictConfig
from torchsummary import summary
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/MIL/AttentionDeepMIL')
from dataloader_TUH import TUH_full_scan_dataset

sys.path.append('/gpfs/space/home/joonas97/MIL')

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')

CROP_SIZE = 120


def train(model, optimizer, train_loader, loss_function, epoch: int,
          check: bool = False):
    model.train()
    train_loss = 0.

    train_count = 0
    train_num_correct = 0

    step = 0
    tepoch = tqdm(train_loader, unit="batch", ascii=True)
    for data, label, file_path in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        step += 1

        data = torch.unsqueeze(data, 0)
        data = torch.permute(data, (0, 1, 4, 2, 3))

        if label == True:
            label = torch.tensor([[0, 1]])
        else:
            label = torch.tensor([[1, 0]])

        data, label = data.cuda(), label.cuda()

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        with torch.cuda.amp.autocast():

            output = model.forward(data)

            loss = loss_function(output, label.float())
            train_loss += loss.item()

            value = torch.eq(output.argmax(dim=1), label.argmax(dim=1))
            train_count += len(value)
            train_num_correct += value.sum().item()

        # backward pass
        loss.backward()
        # step
        optimizer.step()

        if step >= 20 and check:
            break

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_accuracy = train_num_correct / train_count

    print('Train loss: {:.4f}, Train accuracy {:.4f}'.format(train_loss, train_accuracy))
    return train_loss, train_accuracy


def validation(model, test_loader, loss_function, epoch: int,
               check: bool = False):
    model.train()  # keep it in train mode because of a bug with efficient net 3d implementation https://github.com/shijianjian/EfficientNet-PyTorch-3D/issues/6

    test_loss = 0.
    test_error = 0.
    step = 0

    test_count = 0
    test_num_correct = 0

    tepoch = tqdm(test_loader, unit="batch", ascii=True)
    for data, label, file_path in tepoch:
        tepoch.set_description(f"Validation of epoch {epoch}")
        step += 1

        data = torch.unsqueeze(data, 0)
        data = torch.permute(data, (0, 1, 4, 2, 3))

        if label == True:
            label = torch.tensor([[0, 1]])
        else:
            label = torch.tensor([[1, 0]])
        data, label = data.cuda(), label.cuda()

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model.forward(data)

            loss = loss_function(output, label.float())
            test_loss += loss.item()

            value = torch.eq(output.argmax(dim=1), label.argmax(dim=1))
            test_count += len(value)
            test_num_correct += value.sum().item()

        if step >= 20 and check:
            break

    test_accuracy = test_num_correct / test_count
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test accuracy: {:.4f}'.format(test_loss, test_accuracy))
    return test_loss, test_accuracy


@hydra.main(config_path="config", config_name="config_3d", version_base='1.1')
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

    transforms_train = Compose(
        [RandAxisFlip(prob=0.5), RandAxisFlip(prob=0.5),
         RandRotate(range_x=10, prob=1),
         RandGaussianNoise(prob=0.3, mean=0, std=0.2),
         RandCoarseDropout(prob=0.3, holes=40, spatial_size=30)])

    train_dataset = TUH_full_scan_dataset(datasets=cfg.data.datasets, dataset_type="train",
                                          augmentations=transforms_train if cfg.data.data_augmentations is True else None,
                                          plane=cfg.data.plane, center_crop=cfg.data.crop_size,
                                          artificial_spheres=cfg.data.artifical_spheres, downsample=cfg.data.downsample)
    test_dataset = TUH_full_scan_dataset(datasets=cfg.data.datasets, dataset_type="test", plane=cfg.data.plane,
                                         center_crop=cfg.data.crop_size,
                                         artificial_spheres=cfg.data.artifical_spheres, downsample=cfg.data.downsample)
    train_loader = data_utils.DataLoader(train_dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         **loader_kwargs)

    test_loader = data_utils.DataLoader(test_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)

    # read in statistics about the scans for validation metrics
    # base_path = '/gpfs/space/home/joonas97/MIL/AttentionDeepMIL/'
    # test_ROIS = pd.read_csv(base_path + "ROIS/{0}_test_ROIS.csv".format(cfg.data.plane))
    # train_ROIS = pd.read_csv(base_path + "ROIS/{0}_train_ROIS.csv".format(cfg.data.plane))

    logging.info('Init Model')

    model = monai.networks.nets.EfficientNetBN(cfg.model.name, spatial_dims=3, in_channels=1,
                                               num_classes=cfg.model.classes)  # .to(device)
    # if cfg.model.pretrained_medicalnet:
    #     model, inside = create_pretrained_medical_resnet(
    #         pretrained_path=os.path.join('/gpfs/space/home/joonas97', cfg.checkpoint),
    #         model_constructor=resnet34 if cfg.model.name == "resnet34" else resnet18, shortcut_type='A')
    #
    #     # if you need to continue training
    # if "checkpoint" in cfg.keys() and not cfg.model.pretrained_medicalnet:
    #     print("Using checkpoint", cfg.checkpoint)
    #     model.load_state_dict(
    #         torch.load(os.path.join('/gpfs/space/home/joonas97/MIL/AttentionDeepMIL/', cfg.checkpoint)))
    #
    # elif cfg.model.name == "resnet18":
    #     model = monai.networks.nets.resnet18(n_input_channels=1, num_classes=cfg.model.classes)
    # elif cfg.model.name == "resnet34":
    #     model = monai.networks.nets.resnet34(n_input_channels=1, num_classes=cfg.model.classes)
    #
    # if "checkpoint" in cfg.keys() and not cfg.model.pretrained_medicalnet:
    #     print("Using checkpoint", cfg.checkpoint)
    #     model.load_state_dict(torch.load(os.path.join('/gpfs/space/home/joonas97', cfg.checkpoint)))

    if torch.cuda.is_available():
        model.cuda()

    summary(model, (1, 500, 256, 256))

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.999),
                           weight_decay=cfg.training.weight_decay)
    loss_function = torch.nn.CrossEntropyLoss().cuda()

    if not cfg.check:
        experiment = wandb.init(
            project='3D classification experiment on the combined dataset of TUH, totalsegmentor and KITS23',
            resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.training.epochs,
                 learning_rate=cfg.training.learning_rate, model_name=cfg.model.name,
                 weight_decay=cfg.training.weight_decay, directory=os.getcwd()))
    logging.info('Start Training')

    best_test_accuracy = 0
    best_epoch = 0
    not_improved_epochs = 0

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_accuracy = train(model, optimizer,
                                           train_loader,
                                           loss_function, epoch,
                                           check=cfg.check)
        test_loss, test_accuracy = validation(model, test_loader,
                                              loss_function, epoch,
                                              check=cfg.check)
        if cfg.check:
            logging.info("Model check completed")
            return
        if test_accuracy > best_test_accuracy:

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            logging.info(f"Best new model at epoch {epoch} (highest test accuracy)!")

            best_test_accuracy = test_accuracy
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
            'train accuracy': train_accuracy,
            'test accuracy': test_accuracy,
            'epoch': epoch})

    torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
    logging.info(f'Last checkpoint! Checkpoint {epoch} saved!')
    logging.info(f"Training completed, best test accuracy: {best_test_accuracy:.4f} at epoch: {best_epoch}")


if __name__ == "__main__":
    main()
