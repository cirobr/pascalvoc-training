"""
Author: cirobr@GitHub
Date: 24-Aug-2026

Template for training vision models.
"""

# arguments
import os
# cudadevice  = int(os.getenv("CUDA_DEVICE", 0))
epochs      = int(os.getenv("EPOCHS", 2))
minibatch   = int(os.getenv("MINIBATCH", 2))
accum_steps = int(os.getenv("ACCUM_STEPS", 1))
debugflag   = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes", "on")

# public modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import v2
from torchvision.tv_tensors import Image, Mask
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.datasets import VOCSegmentation
from torchmetrics import Accuracy, F1Score, JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import sys
from pathlib import Path

# private modules
from torch_segmentation_losses import IoULoss, GeneralizedDiceFocalLoss
from torch_learn import trainEpoch_aux_accum , evaluateEpoch, evaluatePerClassMetric
from torch_learn import EarlyStopping

# custom modules
sys.path.append("../python_libs")
from cityscapes_tools import encode_labels
from custom_functions import set_determinism, last_n_average, compute_class_frequencies, compute_class_weights


# Device configuration
# torch.cuda.set_device(cudadevice)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {dev}')


# folders
try:
    # Preferred: works when run as file (green triangle, debug, terminal, etc.)
    fpfn = Path(__file__).resolve()
except NameError:
    # Fallback for when code is run in interactive window / jupyter-like
    fpfn = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd()

# script_dir  = fpfn.parent   # parent folder
# script_name = fpfn.name;    # filenamename.ext
outputfolder = fpfn.stem    # filename (no extension)

datasetpath = '../dataset/pascalvoc/'

modelspath = './models/' + outputfolder + '/'
if not os.path.exists(modelspath):
    os.makedirs(modelspath)

tblogspath = './tblogs/' + outputfolder + '/'
if not os.path.exists(tblogspath):
    os.makedirs(tblogspath)


# dataset constants
# imagesize = (1024, 2048)     # original size
scalefixed = (512, 1024)
framesize  = (448, 896)      # square:(512, 512)   # 2 by 1: (384×768, 448×896)
classnrs   = np.arange(19)   # 0-18 classes, 255 is ignore class
C = len(classnrs)


# Training transform (with augmentation)
train_transform = v2.Compose([
    # 1. Geometric augmentation (most important for segmentation)
    v2.RandomResizedCrop(
        size=framesize,
        scale=(0.4, 2.0),
        ratio=(0.8, 1.25),
        antialias=True
    ),
    v2.RandomHorizontalFlip(p=0.5),
    
    # 2. Geometric distortions (very effective for urban scenes)
    v2.RandomAffine(
        degrees=15,            # small rotation
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10,
        fill=0
    ),
    v2.RandomPerspective(distortion_scale=0.2, p=0.5),
    
    # 3. Stronger color / photometric augmentation
    v2.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.15
    ),
    v2.RandomGrayscale(p=0.1),
    v2.RandomPosterize(bits=6, p=0.1),      # occasional posterization
    v2.RandomSolarize(threshold=0.5, p=0.1),
    
    # 4. Additional robustness (helps against overfitting)
    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
    
    # 5. Final conversion & normalization
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # DeeplabV3 normalization
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])


# Validation transform (no randomness)
valid_transform = v2.Compose([
    v2.Resize(size=scalefixed, antialias=True),   # or keep original size?
    v2.CenterCrop(size=framesize),
    # Final conversion & normalization
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # DeeplabV3 normalization
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# Wrapper for training 
def wrap_train_transform(i, m):
    # Important: wrap as tv_tensors
    image = Image(i)
    mask  = Mask(m)

    # Apply full transform in sync for image and mask (geometric + color)
    image, mask = train_transform(image, mask)

    # Encode labels and convert to long tensor
    mask = encode_labels(mask)   # map Cityscapes labels from 0-33 to 0-18 classes
    mask = torch.from_numpy(np.array(mask)).long().squeeze(0)

    return image, mask


# Wrapper for validation (no augmentation)
def wrap_valid_transform(i, m):
    # Important: wrap as tv_tensors
    image = Image(i)
    mask  = Mask(m)

    # Apply full transform in sync for image and mask (geometric + color)
    image, mask = valid_transform(image, mask)

    # Encode labels and convert to long tensor
    mask = encode_labels(mask)   # map Cityscapes labels from 0-33 to 0-19 classes
    mask = torch.from_numpy(np.array(mask)).long().squeeze(0)

    return image, mask


# Load PascalVOC dataset
full_dataset = VOCSegmentation(
    root=datasetpath,
    year='2012',
    image_set='train',
    download='false',
    # transform=
    # target_transform=,
    # transforms=wrap_train_transform,   # note: transforms (plural)
)

train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [0.7, 0.3])
assert False

cs, px = compute_class_frequencies(train_dataset, num_classes=C)
# cs = torch.tensor([2.4141e+08, 2.3166e+07, 1.1156e+08, 3.0265e+06, 3.4000e+06,
#                    6.6764e+06, 1.1329e+06, 3.0751e+06, 9.0490e+07, 4.9606e+06,
#                    2.6375e+07, 7.5897e+06, 1.0336e+06, 3.5327e+07, 1.4770e+06,
#                    1.6057e+06, 1.4516e+06, 5.2571e+05, 2.2060e+06])
# px = 566_483_088

ws = compute_class_weights(cs, total_pixels=px, num_classes=C)
train_weights = ws['median'].to(dev)

# For quick testing
if debugflag:
    epochs        = 2
    minibatch     = 2   # bn does not work with 1
    accum_steps   = 1
    train_dataset = Subset(train_dataset, range(0, 10))
    valid_dataset = Subset(valid_dataset, range(0, 10))
    print("INFO: Debug mode enabled: Using subset of datasets.")
print(f"INFO: Dataset sizes - Training: {len(train_dataset)}, Validation: {len(valid_dataset)}\n")

# DataLoaders
train_loader = DataLoader(train_dataset,
                          batch_size=minibatch,
                          shuffle=True,
                          num_workers=4,
                          drop_last=False,
)

valid_loader = DataLoader(valid_dataset,
                          batch_size=(minibatch * accum_steps),  # 8
                          shuffle=False, 
                          num_workers=4, 
                          drop_last=False,
)


# Load pretrained DeepLabV3/PascalVOC weights (21 classes)
weights = 'DeepLabV3_ResNet101_Weights.DEFAULT'
model = deeplabv3_resnet101(weights=weights, num_classes=21, aux_loss=True).to(dev)

# Adjust model for Cityscapes (19 classes, no background)
in_channels = model.classifier[4].in_channels
model.classifier[4] = nn.Conv2d(in_channels, C, kernel_size=1)  # Adjust final layer
if model.aux_classifier is not None:
    in_aux_channels = model.aux_classifier[4].in_channels
    model.aux_classifier[4] = nn.Conv2d(in_aux_channels, C, kernel_size=1)

# Load pretrained weights
# model_path = './models/trainsum5/bestmodel.pth'
# if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path, map_location=dev))
#     print(f"INFO: Loaded model from {model_path}")
# else:
#     print(f"WARNING: Model not found at {model_path}")

model = model.to(dev)

# Freeze backbone (encoder) layers
# model.backbone.requires_grad_(False)
# print("INFO: Backbone layers frozen. Only classifier and auxiliary classifier will be trained.")


# Loss functions
lossFunction = GeneralizedDiceFocalLoss(
    ignore_index=255,
    reduction="sum",
    focal_gamma=2.0,
    dice_w_type="simple",
    weights=train_weights,
    gdl_weight=1.0,
    focal_weight=1.0,
).to(dev)

validLossFunction = IoULoss(ignore_index=255, reduction="sum").to(dev)


# Optimizer
learning_rate       = 1e-3
final_learning_rate = 1e-4
weight_decay        = 1e-4     # or 1e-4 ~ 5e-4

optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# Scheduler
T1 = 200
sch1 = CosineAnnealingLR(
    optimizer,
    T_max=T1,
    eta_min=final_learning_rate,
)

T2 = 500 - T1
factor = final_learning_rate / learning_rate
sch2 = ConstantLR(optimizer, factor=factor, total_iters=T2)

scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[sch1, sch2],
    milestones=[T1],   # switch to sch2 after T1 epochs
)

# Metrics
accuracy_metric = Accuracy(task="multiclass", num_classes=C, ignore_index=255).to(dev)
accuracy_metric.__name__ = "accuracy_metric"

f1_metric       = F1Score(task="multiclass", num_classes=C, ignore_index=255).to(dev)
f1_metric.__name__ = "f1_metric"

jaccard_metric  = JaccardIndex(task="multiclass", num_classes=C, ignore_index=255).to(dev)
jaccard_metric.__name__ = "jaccard_metric"   

metrics = [accuracy_metric, f1_metric, jaccard_metric]   # single-value metrics only

per_class_iou_metric = MulticlassJaccardIndex(           # multi-value metric
    num_classes=C, average=None, ignore_index=255
).to(dev)
per_class_iou_metric.__name__ = "per_class_iou"


# TensorBoard writer
tblogs = SummaryWriter(tblogspath)


# Early stopping configuration
early_stopping = EarlyStopping(
    number_since_best=20,
    patience=5,
    min_delta=1e-6,
    gap_ratio=float('inf'),   # 1.5,
)

model_saving = EarlyStopping(
    number_since_best=10*epochs,   # to eliminate unwanted messages
    min_delta=1e-6,
    gap_ratio=float('inf'),        # to eliminate unwanted messages
)


# Training loop
validlosses = []
for epoch in range(1, epochs + 1):
    # data loading
    print(f'*** Epoch [{epoch}/{epochs}] ***')
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.3e}')

    # standard training
    # train_loss = libtorch.trainEpoch(
    #     lossFunction, model, train_loader, optimizer,
    #     use_amp=True
    # )

    # training with aux head
    # train_loss = trainEpoch_aux(
    #     lossFunction, model, train_loader, optimizer,
    #     use_amp=True
    # )

    # training with gradient accumulation
    train_loss = trainEpoch_aux_accum(
        lossFunction, model, train_loader, optimizer, 
        accum_steps=accum_steps, use_amp=True
    )

    print(f'Training Loss: {train_loss:.4f}\n')

    # validation
    losses = evaluateEpoch(validLossFunction, model, valid_loader, metrics=metrics)

    valid_loss = losses['loss']
    valid_accuracy = losses['accuracy_metric']    # TensorBoard logging
    tblogs.add_scalar('Loss/Training', train_loss, epoch)
    tblogs.add_scalar('Loss/Validation', valid_loss, epoch)

    valid_f1       = losses['f1_metric']
    valid_jaccard  = losses['jaccard_metric']

    validlosses.append(valid_loss)
    avg_valid_loss = last_n_average(validlosses, 3)

    print(f'Validation Loss: {valid_loss:.4f}')
    print(f'  Avg val loss : {avg_valid_loss:.4f}')
    print(f'  Accuracy : {valid_accuracy:.4f}')
    print(f'  F1-Score : {valid_f1:.4f}')
    print(f'  Jaccard-Score : {valid_jaccard:.4f}\n')

    # per-class IoU evaluation
    valid_per_class_iou = \
        evaluatePerClassMetric(model, valid_loader, per_class_iou_metric)  # tensor of shape [C]
    
    print('  Per-class IoU:')
    for class_idx in range(C):
        print(f'    Class {class_idx}: IoU = {valid_per_class_iou[class_idx].item():.4f}')
    print()

    # TensorBoard logging
    tblogs.add_scalar('Loss/Training', train_loss, epoch)
    tblogs.add_scalar('Loss/Validation', valid_loss, epoch)
    tblogs.add_scalar('Loss/Avg-Val-Loss', avg_valid_loss, epoch)
    tblogs.add_scalar('Metric/Accuracy', losses['accuracy_metric'], epoch)
    tblogs.add_scalar('Metric/F1Score', losses['f1_metric'], epoch)
    tblogs.add_scalar('Metric/Jaccard-Score', losses['jaccard_metric'], epoch)
    for class_idx in range(C):
        tblogs.add_scalar(f'Metric/PerClassIoU/Class_{class_idx}', valid_per_class_iou[class_idx].item(), epoch)

    # step the learning rate scheduler after each epoch
    scheduler.step()

    # Early-stop check
    model_saving(train_loss, valid_loss)
    early_stop = early_stopping(train_loss, avg_valid_loss)

    if model_saving.counter_since_best == 0:   # save if model has improved
        torch.save(model.state_dict(), modelspath + 'model.pth')

    if early_stop:
        break

# Close project
tblogs.close()

# Save final model
if os.path.exists(modelspath + 'model.pth'):
    os.rename(modelspath + 'model.pth', modelspath + 'bestmodel.pth')
print("Project completed!")
