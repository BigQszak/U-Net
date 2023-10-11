import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet

import hyperparameters

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def transformation():
    pass


def main():
    # change to functions
    train_transform = A.Compose(
        [
            A.Resize(
                height=hyperparameters.IMAGE_HEIGHT, width=hyperparameters.IMAGE_WIDTH
            ),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=1.0),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(
                height=hyperparameters.IMAGE_HEIGHT, width=hyperparameters.IMAGE_WIDTH
            ),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    model = UNet(in_channels=3, out_channels=1).to(device=DEVICE)

    # Changing out_channels to ex. 3 & loss to cross entropy will yield class segmentation

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        hyperparameters.TRAIN_IMG_DIR,
        hyperparameters.TRAIN_MASK_DIR,
        hyperparameters.VAL_IMG_DIR,
        hyperparameters.VAL_MASK_DIR,
        hyperparameters.BATCH_SIZE,
        train_transform,
        val_transform,
        hyperparameters.NUM_WORKERS,
        hyperparameters.PIN_MEMORY,
    )

    if hyperparameters.LOAD_MODEL:
        load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"), model)

    # check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(hyperparameters.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print examples
        save_predictions_as_imgs(
            val_loader,
            model,
            folder=os.path.join(os.path.dirname(__file__), "predictions"),
            device=DEVICE,
        )


if __name__ == "__main__":
    main()
