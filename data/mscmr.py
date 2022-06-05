import SimpleITK as sitk
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils import data
import nibabel as nib
import os
import data.transforms as T

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

class seg_train(data.Dataset):
    def __init__(self, img_paths, lab_paths, lab_values, transforms):
        self._transforms = transforms
        self.lab_values = lab_values
        self.examples = []
        self.img_dict = {}
        self.lab_dict = {}
        for img_path, lab_path in zip(sorted(img_paths), sorted(lab_paths)):
            img = self.read_image(str(img_path))
            img_name = img_path.stem
            self.img_dict.update({img_name : img})
            lab = self.read_label(str(lab_path))
            lab_name = lab_path.stem
            self.lab_dict.update({lab_name : lab})
            assert img[0].shape[2] == lab[0].shape[2]
            self.examples += [(img_name, lab_name, -1, -1, slice) for slice in range(img[0].shape[2])]
            
    def __getitem__(self, idx):
        img_name, lab_name, Z, X, Y = self.examples[idx]
        if Z != -1:
            img = self.img_dict[img_name][Z, :, :]
            lab = self.lab_dict[lab_name][Z, :, :]
        elif X != -1:
            img = self.img_dict[img_name][:, X, :]
            lab = self.lab_dict[lab_name][:, X, :]
        elif Y != -1:
            img = self.img_dict[img_name][0][:, :, Y]
            scale_vector_img = self.img_dict[img_name][1]
            lab = self.lab_dict[lab_name][0][:, :, Y]
            scale_vector_lab = self.lab_dict[lab_name][1]
        else:
            raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
        img = np.expand_dims(img, 0)
        lab = np.expand_dims(lab, 0)
        target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'orig_size': lab.shape}
        if self._transforms is not None:
            img, target = self._transforms([img, scale_vector_img], [target,scale_vector_lab])
        return img, target

    def read_image(self, img_path):
        img_dat = load_nii(img_path)
        img = img_dat[0]
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        img = img.astype(np.float32)
        return [(img-img.mean())/img.std(), scale_vector]

    def read_label(self, lab_path):
        lab_dat = load_nii(lab_path)
        lab = lab_dat[0]
        pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        # cla = np.asarray([(lab == v)*i for i, v in enumerate(self.lab_values)], np.int32)
        return [lab, scale_vector]

    def __len__(self):
        return len(self.examples)

def make_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize()
    ])

    if image_set == 'train':
        # return T.Compose([T.RandomHorizontalFlip(),T.RandomCrop([256, 256]), T.ToTensor()])
        return T.Compose([
            T.Rescale(),
            T.RandomHorizontalFlip(),
            T.RandomRotate((0,360)),
            # T.CenterRandomCrop([0.7,1]),
            # T.RandomResize([0.8,1.2]), 
            T.PadOrCropToSize([212,212]),
            # T.RandomColorJitter(),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # T.CenterRandomCrop([0.7,1]),
            # T.RandomResize([0.8,1.2]), 
            T.Rescale(),
            T.PadOrCropToSize([212,212]),
            # T.RandomColorJitter(),
            normalize])


    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path('/data/zhangke/datasets/' + args.dataset)
    assert root.exists(), f'provided MSCMR path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels"),
        "val": (root / "val" / "images", root / "val" / "labels"),
    }
    if image_set == "train":
        img_folder, lab_folder = PATHS[image_set]
        dataset_dict_train = {}
        dataset_dict_hold = {}
        for task, value in args.tasks.items():
            lab_values = value['lab_values']
            img_paths = sorted(list(img_folder.iterdir()))
            lab_paths = sorted(list(lab_folder.iterdir()))
            order_indexs = [i for i in range(len(img_paths))]
            random.shuffle(order_indexs)
            img_paths = [img_paths[i] for i in order_indexs]
            lab_paths = [lab_paths[i] for i in order_indexs]
            n = len(img_paths)
            dataset_train = seg_train(img_paths[0:n], lab_paths[0:n], lab_values, transforms=make_transforms(image_set))
            dataset_dict_train.update({task : dataset_train})
            dataset_hold = seg_train(img_paths[int(0.9*n):n], lab_paths[int(0.9*n):n], lab_values, transforms=make_transforms(image_set))
            dataset_dict_hold.update({task : dataset_hold})
            return dataset_dict_train, dataset_dict_hold
    elif image_set=="val":
        img_folder, lab_folder = PATHS[image_set]
        dataset_dict_val = {}
        for task, value in args.tasks.items():
            lab_values = value['lab_values']
            img_paths = sorted(list(img_folder.iterdir()))
            lab_paths = sorted(list(lab_folder.iterdir()))
            dataset_val = seg_train(img_paths, lab_paths, lab_values, transforms=make_transforms(image_set))
            dataset_dict_val.update({task : dataset_val})
            return dataset_dict_val
