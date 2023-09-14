""" 
Description: 
    Creates a dataloader to process raw data.
"""

import os
from pathlib import Path
import logging
import albumentations as A
from torch.utils.data import Dataset, DataLoader
#from torch import nn
import pandas as pd
import cv2 as cv
import torch
import numpy as np

logger = logging.getLogger(__name__)
logger.handlers = []
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s')) # %(asctime)s -
logger.addHandler(sh)

logger.setLevel(logging.CRITICAL)

def make_transformer(aug_hyps, job_type='train'):

    """Create a transformer for data augmentation"""

    assert job_type in [
        'train', 'inference'], 'please use a valid job_type.'

    transformer = A.Compose([

        A.HorizontalFlip(p=aug_hyps['RandomHorizontalFlip']),

        A.ColorJitter(brightness=aug_hyps['ColorJitter'][0],
                      contrast=aug_hyps['ColorJitter'][1],
                      saturation=aug_hyps['ColorJitter'][2],
                      hue=aug_hyps['ColorJitter'][3],
                      p=aug_hyps['ColorJitter'][-1]),

        A.Perspective(scale=aug_hyps['RandomPerspective'][0],
                      keep_size=True,
                      pad_mode=0,
                      pad_val=0,
                      mask_pad_val=0,
                      fit_output=False,
                      interpolation=1,
                      always_apply=False,
                      p=aug_hyps['RandomPerspective'][-1]),


        A.GaussianBlur(blur_limit=(3, 5),  # data_hyps['GaussianBlur'][0]
                       sigma_limit=aug_hyps['GaussianBlur'][1],
                       p=aug_hyps['GaussianBlur'][-1]),

        A.Affine(translate_percent=aug_hyps['RandomAffine'][0],
                 scale=aug_hyps['RandomAffine'][1],
                 rotate=aug_hyps['RandomAffine'][2],
                 fit_output=True,
                 mode=3,
                 p=aug_hyps['RandomAffine'][-1]),

        A.Sharpen(alpha=aug_hyps['RandomAdjustSharpness'][0],
                  lightness=aug_hyps['RandomAdjustSharpness'][1],
                  p=aug_hyps['RandomAdjustSharpness'][-1]),

        A.Resize(aug_hyps['Resize'][0], aug_hyps['Resize'][0], interpolation=aug_hyps['Resize'][1]), # 1:cv2.INTER_LINEAR (BILINEAR), 2:cv.INTER_CUBIC (BICUBIC)

        # Normalize using ImageNet mean and std.
        A.Normalize(mean=aug_hyps['Normalize'][0], std= aug_hyps['Normalize'][1], p=aug_hyps['Normalize'][-1])

        # mode 0,1,2..
        # cv2.BORDER_CONSTANT
        # cv2.BORDER_REFLECT
        # cv2.BORDER_REFLECT_101
        # cv2.BORDER_REPLICATE
        # cv2.BORDER_WRAP

    ], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.5,
                                label_fields=['class_labels'], check_each_transform=False))

    # inference transformer

    inference_transformer = A.Compose([

        # to match resnet expected input.
        # https://pytorch.org/hub/pytorch_vision_resnet/

        A.Resize(aug_hyps['Resize'][0], aug_hyps['Resize'][0], interpolation=aug_hyps['Resize'][1]), # 1:cv2.INTER_LINEAR (BILINEAR), 2:cv.INTER_CUBIC (BICUBIC)

        # Normalize using ImageNet mean and std.
        A.Normalize(mean=aug_hyps['Normalize'][0], std= aug_hyps['Normalize'][1], p=aug_hyps['Normalize'][-1])

    ], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.5,
                                 label_fields=['class_labels'], check_each_transform=False))
    
    visualization_transformer = A.Compose([

        A.Resize(aug_hyps['Resize'][0], aug_hyps['Resize'][0], interpolation=aug_hyps['Resize'][1]), # 1:cv2.INTER_LINEAR (BILINEAR), 2:cv.INTER_CUBIC (BICUBIC)

    ], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.5,
                                 label_fields=['class_labels'], check_each_transform=False))

    if job_type == 'train':
        return transformer
    if job_type == 'inference':
        return inference_transformer
    if job_type == 'visualization':
        return visualization_transformer

# batch_size=1, num_workers=0, crop=False, zoom_out=0, output_size=160):

class DataloaderCsv(Dataset):
    """
    > Description
        _summary_

    > Args:
        csv_file {str}: cvs file with image path and class.
        root_dir {str}: directory where the data is stored.
        transform {}: transformations.
    """

    def __init__(self, csv_file: str, root_dir: str,  transform=None, 
                 job_type='multiclass'):

        self.data = pd.read_csv(csv_file) 
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.job_type = job_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # * Loading the image and label.
        # row i, column 0.
        img_path = os.path.join(self.root_dir, Path(self.data.iloc[index, 0]))

        # loading image and converting to RGB.
        # albumentations need numpy format.
        # ! TODO if possible, convert image to RGB only in transformation.
        # ! TODO specify model transformations in another file.
        # re scaling to [0,1] to match resnet input format
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
        image = (cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB) / 255.0).astype(np.float32)
        logger.debug(f'{image.dtype}, should be float32') 
        image_name = Path(img_path).stem
        
        match self.job_type:
            case 'multilabel':
                label = np.array(self.data.iloc[index, 1:].values, dtype=np.float32) # if one hot load a vector of labels. # * Multi label
                label = torch.tensor(label, dtype = torch.float32) # * Multi label

                logger.debug(f'labels: {label}, {type(label)}, {label.dtype}')

            case 'multiclass':
                label = int(self.data.iloc[index, 1]) # * Multi class
                label = torch.as_tensor(label, dtype = torch.int64) # * Multi class
            case _:
                raise ValueError("Undefined job type")

        # * get original image size, aspect ratio and resolution.
        img_size = image.shape
        width = img_size[1]
        height = img_size[0]
        # width/height
        aspect_ratio = round(width/height, 4)
        resolution = width * height

        bbox = (0.5, 0.5, 1, 1)

        logger.debug('image | type: %s, dtype: %s, size: %s, max: #s',  \
                     type(image), image.dtype, image.size, [image[..., 0].max(),image[..., 1].max(),image[..., 2].max()])

        # * apply transforms (CPU).
        # ! TODO measure difference between CPU and GPU time for transformation.
        # ! (possible to transform into GPU?)
        # ! albumentations.augmentations.transforms.transforms.ToTensorV2
        if self.transform:
            # apply albumentations (transformations in the image and it's labels (bbox, id))
            aug_data = self.transform(
                image=image, bboxes=[bbox], class_labels=[label])

            # results of transformation.
            image = aug_data['image']
            bbox = aug_data['bboxes']

        # channels, width, height.
        # torch.tensor always copies data while torch.as_tensor() avoids copying data if possible.
        image = torch.as_tensor(image, dtype=torch.float32).permute(
            2, 0, 1)  # .to(self.device)


        #logger.debug(f'image: type: {type(image)} dtype: {image.dtype}, size: {image.size}')
        #logger.debug(f'label: type: {type(label)} dtype: {label.dtype}, size: {label.size}')
        logger.debug('image | type: %s, dtype: %s, size: %s', type(image), image.dtype, image.size)
        logger.debug('label | type: %s, dtype: %s, size: %s', type(label), label.dtype, label.size)

        return {'images': image, 'labels': label,
                'original_width': width, 'original_height': height,
                'aspect_ratio': aspect_ratio,'resolution': resolution, 
                'name': image_name}

def batch_loader(csv_file, root_dir, transform=None, batch_size=16,
                 num_workers=1, shuffle=False, generator=None, worker_init_fn=None, job_type = 'multiclass'):

    """Integrate the loader with the Dataloader for batch processing"""

    loader = DataloaderCsv(
        csv_file=csv_file, root_dir=root_dir,  transform=transform, job_type=job_type)

    # persistent_workers: creating the workers is expensive, this will keep the workers, 
    # disadvantage: will use more RAM.
    return DataLoader(dataset=loader,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      generator=generator,
                      worker_init_fn=worker_init_fn,
                      pin_memory=True,
                      persistent_workers=True if num_workers > 0 else False) 
