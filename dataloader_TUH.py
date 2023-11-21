import glob
import os
import random
from typing import List, Tuple

import nibabel as nib
import numpy as np
import raster_geometry as rg
import torch
import torch.nn.functional as F
from monai.transforms import *


def threshold_at_one(x):
    # threshold at 1
    return x > -1


background_cropper = CropForeground(select_fn=threshold_at_one)

# TODO: change spatial cropper for other plane types !!
transforms_train = Compose(
    [
        RandRotate(range_x=1, prob=1),
        RandGaussianNoise(prob=1, mean=0, std=0.2),
        RandAffine(prob=0.5, scale_range=(-0.1, 0.1), translate_range=(-50, 50), rotate_range=(-1, 1),
                   padding_mode="border")])


def add_random_sphere(image):
    size = image.shape[1]
    height = image.shape[2]

    z, y, x = np.random.uniform(0.1, 0.9), np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)

    radius = np.random.randint(20, 40)

    sphere_mask = rg.sphere((size, size, height), radius, (y, x, z))
    gaussian_noise = torch.HalfTensor(np.random.randn(size, size, height) * 0.2 + 1.7)

    image[sphere_mask] = gaussian_noise[sphere_mask]

    return image


def get_dataset_paths(datasets: List[str], dataset_type: str) -> Tuple[List[str], List[str], int]:
    ''' currently possible dataset types: TUH_kidney, totalsegmentor
    dataset type is either train or test'''

    TUH_data_path = '/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/new_setup/MIL_EXP/'
    total_segmentor_data_path = '/gpfs/space/projects/BetterMedicine/joonas/totalsegmentor/model_ready_dataset/'
    kits23_data_path = '/gpfs/space/projects/BetterMedicine/joonas/kits23/model_ready_dataset/'

    all_controls = []
    all_tumors = []
    print("-------------------------")
    print("DATASET TYPE (TRAIN/TEST):", dataset_type)
    print("using the combination of ", len(datasets), "datasets")

    TUH_study_length = 1  # used for calculating attention accuracy, if no TUH is used, will be 1
    if "TUH_kidney" in datasets:
        data_path = os.path.join(TUH_data_path, dataset_type)

        control_path = data_path + '/controls2/*nii.gz'
        tumor_path = data_path + '/tumors2/*nii.gz'
        print("PATHS TUH:")
        print(control_path)
        print(tumor_path)
        control = glob.glob(control_path)
        tumor = glob.glob(tumor_path)

        all_controls.extend(control)
        all_tumors.extend(tumor)

        TUH_study_length = len(control) + len(tumor)
    if "totalsegmentor" in datasets:
        data_path = os.path.join(total_segmentor_data_path, dataset_type)

        control_path = data_path + '/control/*nii.gz'
        tumor_path = data_path + '/tumor/*nii.gz'
        cyst_path = data_path + '/cyst/*nii.gz'

        print("PATHS totalsegmentor:")
        print(control_path)
        print(tumor_path)
        print(cyst_path)

        control = glob.glob(control_path)
        tumor = glob.glob(tumor_path)
        cyst = glob.glob(cyst_path)

        all_controls.extend(control)
        all_controls.extend(cyst)  # also use cysts as controls, might alter in the future
        all_tumors.extend(tumor)

    if "kits23" in datasets:
        data_path = os.path.join(kits23_data_path, dataset_type)
        data_path = data_path + "/*nii.gz"
        print("PATH kits23:")
        print(data_path)
        # kits only has tumor cases
        tumor = glob.glob(data_path)
        # take only fraction of kits to keep dataset class balance
        tumor = tumor[:int(len(tumor) * 0.37)]
        all_tumors.extend(tumor)

    return all_controls, all_tumors, TUH_study_length


class TUH_full_scan_dataset(torch.utils.data.Dataset):
    def __init__(self, datasets: List[str], dataset_type: str, only_every_nth_slice: int = 1,
                 downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 min_max_normalization: bool = False, sample_shifting: bool = False, plane: str = 'axial',center_crop: int = 120,artificial_spheres : bool = False):
        super(TUH_full_scan_dataset, self).__init__()
        self.as_rgb = as_rgb
        self.min_max_norm = min_max_normalization
        self.augmentations = augmentations
        self.nth_slice = only_every_nth_slice
        self.downsample = downsample
        self.sample_shifting = sample_shifting
        self.plane = plane
        self.artifical_spheres = artificial_spheres

        self.center_cropper = CenterSpatialCrop(roi_size=(512, 512, center_crop))  # 500
        print("PLANE: ", plane)
        print("CROP SIZE: ", center_crop)
        print("ARTIFICIAL SPHERES EXPERIMENT: ", artificial_spheres)
        control, tumor, TUH_length = get_dataset_paths(datasets=datasets, dataset_type=dataset_type)
        control_labels = [[False]] * len(control)
        tumor_labels = [[True]] * len(tumor)
        self.TUH_length = TUH_length
        self.img_paths = control + tumor
        self.labels = control_labels + tumor_labels

        print("Data length: ", len(self.img_paths), "Label length: ", len(self.labels))
        print(
            f"control: {len(control)}, tumor: {len(tumor)}")

        self.classes = ["control", "tumor"]

    def __len__(self):
        # a DataSet must know its size
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]

        x = nib.load(path).get_fdata()

        # PLANES: set it into default plane (axial)
        # if transformations are needed we start from this position
        if not "kits23" in path:
            x = np.flip(x, axis=1)
            x = np.transpose(x, (1, 0, 2))
        else:  # kits is in another orientation
            x = np.transpose(x, (1, 2, 0))
        # this should give the most common axial representation: (patient on their back)
        if self.plane == "axial":
            pass
            # originally already in axial format
        elif self.plane == "coronal":
            x = np.transpose(x, (2, 1, 0))
        elif self.plane == "sagital":
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError('plane is not correctly specified')


        h, w, d = x.shape

        if self.downsample:
            x = torch.from_numpy(x.copy())
            x = F.interpolate(torch.unsqueeze(torch.unsqueeze(x,0),0), scale_factor=0.5,
                                   mode='trilinear', align_corners=False)


            x = np.array(torch.squeeze(x))
        if not self.sample_shifting:

            if x.shape[2] < 80 and "tuh_kidney" not in path:
                # take more slices if the scan is small, currently cannot do it for tuh study as attention accuracy is measured only on certain slices
                x = x[:, :, ::max(int(self.nth_slice - 2), 1)]
            else:
                x = x[:, :, ::self.nth_slice]
        else:
            shift = random.randint(0, self.nth_slice - 1)
            x = x[:, :, shift::self.nth_slice]

        clipped_x = np.clip(x, np.percentile(x, q=0.05), np.percentile(x, q=99.5))
        norm_x = (clipped_x - np.mean(clipped_x, axis=(0, 1))) / (
                np.std(clipped_x, axis=(0, 1)) + 1)  # mean 0, std 1 norm

        norm_x = torch.unsqueeze(torch.from_numpy(norm_x), 0)
        norm_x = self.center_cropper(norm_x)


        norm_x = torch.squeeze(norm_x)

        x = norm_x.to(torch.float16)


        if self.artifical_spheres:

            label = np.random.randint(0, 2)

            if label == 1:
                #create sphere
                x = add_random_sphere(x)
                y = torch.tensor([True])
            else:
                # do not create sphere
                y = torch.tensor([False])
        else:
            y = torch.tensor(self.labels[index])

        if self.augmentations is not None:
            for i in range(x.shape[2]):
                x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))

        if self.as_rgb:
            x = torch.stack([x, x, x], dim=0)
            x = torch.squeeze(x)

        return x, y, self.img_paths[index]
