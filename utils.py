import os
import cv2
import time
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from torchvision.utils import make_grid
from datetime import datetime

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print("Dataset already exists on the disk. Skipping download.")
        return
    
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(filepath)) as t: 
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n

def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    shutil.unpack_archive(filepath, extract_dir)

def get_filenames(dataset_directory):
    filepath = os.path.join(dataset_directory, "images.tar.gz")
    download_url(
        url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", filepath=filepath
    )
    extract_archive(filepath)

    filepath = os.path.join(dataset_directory, "annotations.tar.gz")
    download_url(
        url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", filepath=filepath,
    )
    extract_archive(filepath)

    root_directory = os.path.join(dataset_directory)
    images_directory = os.path.join(root_directory, "images")
    masks_directory = os.path.join(root_directory, "annotations", "trimaps")

    images_filenames = list(sorted(os.listdir(images_directory)))

    print("Eliminating corrupted data.")
    correct_images_filenames = [i for i in images_filenames if i not in CORRUPTED_IMAGES_FILENAMES and cv2.imread(os.path.join(images_directory, i)) is not None]

    random.seed(42)
    random.shuffle(correct_images_filenames)

    train_images_filenames = np.array(correct_images_filenames[:6000])
    val_images_filenames = np.array(correct_images_filenames[6000:])

    return train_images_filenames, val_images_filenames, images_directory, masks_directory

def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask

def get_data(data, filenames, images_directory, masks_directory, args):
    height, width = args["size"]
    transform = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=(0.4816, 0.4495, 0.3961), std=(0.2669, 0.219, 0.2707)),
        ToTensorV2()
    ]) 

    for i, filename in enumerate(filenames):
        if args["num_classes"] == 1: # binary segmentation
            data["labels"][i] = 0
        else:
            data["labels"][i] = 0 if filename[0].isupper() else 1
        
        image = cv2.imread(os.path.join(images_directory, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            os.path.join(masks_directory, filename.replace(".jpg", ".png")),
            cv2.IMREAD_UNCHANGED,
        )
        mask = preprocess_mask(mask)

        transformed = transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        data["images"][i] = image
        data["masks"][i] = mask
    
    return data

class OxfordPetDataset(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)
    
    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            os.path.join(self.masks_directory, image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED,
        )
        mask = preprocess_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return image.float(), mask.float()

class TensorDataset(Dataset):
    def __init__(self, syn_images, args, soft_label_training=False):
        self.images = syn_images[:, :3, :, :]
        self.masks = syn_images[:, 3:, :, :]
        
        if args["soft_label"] == "clamp" and soft_label_training:
            self.masks = torch.clamp(self.masks, 0, 1)

        if args["soft_label"] == "sigmoid" and soft_label_training:
            self.masks = self.masks.sigmoid()

        self.augmentation = CustomAugmentation(args)
        self.args = args

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx].squeeze(0)
        image, mask = self.augmentation.apply_augmentation(image, mask)
        return image, mask

class CustomAugmentation:
    def __init__(self, args):
        self.args = args

    def apply_augmentation(self, image, mask):
        strategy = random.choice(self.args["eval_train_strategy"].split('_'))

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        if "rotate" in strategy:
            angle = random.uniform(-30, 30)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)

        if "scale" in strategy:
            scale_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.affine(image, angle=0, translate=(0, 0), scale=scale_factor, shear=0)
            mask = transforms.functional.affine(mask, angle=0, translate=(0, 0), scale=scale_factor, shear=0)

        if "flip" in strategy:
            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

        image = image.squeeze(0)
        mask = mask.squeeze(0)

        return image, mask


def get_val_dataset_from_filenames(val_images_filenames, images_directory, masks_directory, transform):
    return OxfordPetDataset(val_images_filenames, images_directory, masks_directory, transform)

def get_dataloader_from_filenames(val_images_filenames, images_directory, masks_directory, transform, batch_size=256, shuffle=True):
    dataset = OxfordPetDataset(val_images_filenames, images_directory, masks_directory, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

def get_syn_loader_from_tensors(syn_images, soft_label_training, args):
    syn_dataset = TensorDataset(syn_images, args, soft_label_training)
    return DataLoader(syn_dataset, batch_size=args["eval_batch_size"], shuffle=True, pin_memory=True)

def train(train_loader, model, criterion, optimizer, epoch, args, retval_model=False):
    '''
        training 
    '''
    total_train_loss = 0
    total_dice_score = 0
    num_samples = 0

    model.train()
    train_loop = tqdm(train_loader, desc=f"Epoch: {epoch} - Training  ")
    for i, (images, target) in enumerate(train_loop, start=1):
        batch_size = images.size(0)
        images = images.to(args["device"], non_blocking=True)
        target = target.to(args["device"], non_blocking=True)

        output = model(images).squeeze(1)

        if args.get("loss_func", "bce") == "bce":
            output = output.sigmoid()
        else:
            if target.ndim == 4 and target.shape[1] == 2:
                loss = criterion(output, target)
            else:
                target_class_probs = prepare_mask_class_probabilities(target)
                loss = criterion(output, target_class_probs)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args["loss_func"] == "ce":
            dice_score = dice_coefficient(output, target, args)
        else: 
            dice_score = dice_coefficient(output, target, args)

        total_train_loss += loss.item() * batch_size
        total_dice_score += dice_score.item()
        num_samples += batch_size

        train_loop.set_postfix(loss=total_train_loss/num_samples, dice_score=total_dice_score/i)

    if retval_model:
        return model

    avg_train_loss = total_train_loss / num_samples
    avg_dice_score = total_dice_score / i
    return avg_train_loss, avg_dice_score

def validate(val_loader, model, criterion, epoch, args):
    '''
        validate
    '''
    total_val_loss = 0
    total_dice_score = 0
    num_samples = 0

    model.eval()
    val_loop = tqdm(val_loader, desc=f"Epoch: {epoch} - Validation")
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loop, start=1):
            batch_size = images.size(0)
            images = images.to(args["device"], non_blocking=True)
            target = target.to(args["device"], non_blocking=True)

            output = model(images).squeeze(1)

            if args.get("loss_func", "bce") == "bce":
                output = output.sigmoid()
                loss = criterion(output, target)
            else: 
                if target.shape[1] == 2:            
                    loss = criterion(output, target)
                else:
                    target_class_probs = prepare_mask_class_probabilities(target)
                    loss = criterion(output, target_class_probs)
            if args["loss_func"] == "ce":
                dice_score = dice_coefficient(output, target_class_probs, args)
            else: 
                dice_score = dice_coefficient(output, target, args)

            total_val_loss += loss.item() * batch_size
            total_dice_score += dice_score.item()
            num_samples += batch_size

            val_loop.set_postfix(loss=total_val_loss/num_samples, dice_score=total_dice_score/i)

    avg_val_loss = total_val_loss / num_samples
    avg_dice_score = total_dice_score / i

    return avg_val_loss, avg_dice_score

def prepare_mask_class_probabilities(batch_masks):
    """
    Prepare a batch of 128x128 binary masks of 0s and 1s to be used as targets for CrossEntropyLoss.
    This version is compatible with a 4D tensor input of shape [batch_size, 1, 128, 128].

    Args:
    - batch_masks: A 4D tensor or array of shape [batch_size, 1, 128, 128] with values 0.0 or 1.0.

    Returns:
    - prepared_masks: A 4D tensor of shape [batch_size, 2, 128, 128] with class probabilities for 0 and 1.
    """
    if not isinstance(batch_masks, torch.Tensor):
        batch_masks = torch.tensor(batch_masks)

    # Ensure the mask is in the right shape (batch_size, 128, 128)
    if batch_masks.ndim == 4 and batch_masks.shape[1] == 1:
        batch_masks = batch_masks.squeeze(1)

    # Convert the mask to a LongTensor, which is required for CrossEntropyLoss targets
    # Create class probability targets
    class_0_probs = 1 - batch_masks
    class_1_probs = batch_masks

    # Stack along a new dimension to create a 4D tensor
    prob_targets = torch.stack([class_0_probs, class_1_probs], dim=1)

    return prob_targets

class EarlyStopping:
    def __init__(self, patience=25, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def dice_coefficient(pred, target, args, epsilon=1e-7):
    """
    Compute the Dice coefficient.

    Args:
    - pred: Predictions from the network (before sigmoid).
    - target: Ground truth labels.
    - epsilon: Small constant to prevent division by zero.

    Returns:
    - dice: Dice coefficient.
    """

    loss_func = args.get("loss_func", "bce")

    if loss_func == "bce":
        pred = pred.sigmoid()

    if (pred.ndim == 4 and pred.shape[1] == 2) and (target.ndim == 4 and target.shape[1] == 1):
        pred = F.softmax(pred, dim=1)

    if (pred.ndim == 4 and pred.shape[1] == 2) and (target.ndim == 4 and target.shape[1] == 2):
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        target = torch.argmax(target, dim=1)

    # Flatten the tensors. This works for both 2D and 3D images and batches of images.
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Compute the intersection and the sum of the two sets.
    # Here we avoid explicitly binarizing the prediction. 
    intersect = (pred * target).sum()
    denominator = pred.sum() + target.sum()
    
    dice = (2. * intersect + epsilon) / (denominator + epsilon)
    
    return dice

def plot_image_mask_pairs(syn_images, mean, std, args):
    images = syn_images[:, :3, :, :]
    masks = syn_images[:, 3:, :, :]

    if args["loss_func"] == "ce":
        masks = masks[:, 1, :, :].unsqueeze(1)

    images = images * std.view(3, 1, 1) + mean.view(3, 1, 1)

    images_grid = make_grid(images, nrow=10)
    masks_grid = make_grid(masks, nrow=10)

    if args["ipc"] == 10:
        figsize=(40, 5)
    else: 
        figsize=(40, 40)

    # Plotting the grids
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    
    # Plot cat images and masks
    axs[0].imshow(images_grid.permute(1, 2, 0))
    axs[0].axis('off')

    # Plot dog images and masks
    axs[1].imshow(masks_grid.permute(1, 2, 0).squeeze(-1), cmap='gray')
    axs[1].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()

def save_image_mask_pairs(syn_images, mean, std, args):
    os.makedirs(args["save_dir"], exist_ok=True)

    images = syn_images[:, :3, :, :]
    masks = syn_images[:, 3:, :, :]

    if args["loss_func"] == "ce":
        masks = masks[:, 1, :, :].unsqueeze(1)

    images = images * std.view(3, 1, 1) + mean.view(3, 1, 1)

    images_grid = make_grid(images, nrow=10)
    masks_grid = make_grid(masks, nrow=10)

    if args["ipc"] == 10:
        figsize=(40, 5)
    else: 
        figsize=(40, 40)

    fig, axs = plt.subplots(2, 1, figsize=figsize)

    axs[0].imshow(images_grid.permute(1, 2, 0))
    axs[0].set_aspect('equal')
    axs[0].axis('off')
    
    axs[1].imshow(masks_grid.permute(1, 2, 0).squeeze(-1), cmap='gray')
    axs[1].set_aspect('equal')
    axs[1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(args["save_dir"], f"figure_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"))
    plt.close()

def preprocess_experts(experts_directory):
    print("Preprocessing expert trajectories...")
    experts_filenames = {}
    for expert_filename in sorted(os.listdir(experts_directory)):
        if expert_filename.startswith("model"):
            parts = expert_filename.split("_")
            e_num = int(parts[0][5:])
            
            if e_num not in experts_filenames:
                experts_filenames[e_num] = []
            
            experts_filenames[e_num].append(os.path.join(experts_directory, expert_filename))
    print(f"Finished! {max(experts_filenames.keys())} experts available.")
    return experts_filenames

class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param, y=None):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

STANDARD_TRANSFORM = lambda args : A.Compose(
    [
        A.Resize(args["size"][0], args["size"][1]),
        A.Normalize(mean=(0.4816, 0.4495, 0.3961), std=(0.2669, 0.2619, 0.2707)),
        ToTensorV2()
    ]
)

SHIFT_SCALE_ROTATE_TRANSFORM = lambda args : A.Compose(
    [
        A.Resize(args["size"][0], args["size"][1]),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.Normalize(mean=(0.4816, 0.4495, 0.3961), std=(0.2669, 0.2619, 0.2707)),
        ToTensorV2()
    ]
)

CORRUPTED_IMAGES_FILENAMES = {
    # corrupted cats
    'Egyptian_Mau_129.jpg',
    'Egyptian_Mau_162.jpg',
    'Egyptian_Mau_165.jpg',
    'Egyptian_Mau_196.jpg',
    'Egyptian_Mau_20.jpg',
    'Persian_259.jpg',
    # corrupted dogs
    'beagle_116.jpg',
    'chihuahua_121.jpg',
    'japanese_chin_199.jpg',
    'keeshond_7.jpg',
    'leonberger_18.jpg',
    'miniature_pinscher_14.jpg',
    'saint_bernard_108.jpg',
    'saint_bernard_15.jpg',
    'saint_bernard_60.jpg',
    'saint_bernard_78.jpg',
    'staffordshire_bull_terrier_2.jpg',
    'staffordshire_bull_terrier_22.jpg',
    'wheaten_terrier_195.jpg'
}