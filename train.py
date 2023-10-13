import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter  # print to tensorboard

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


def train_fn(loader, model, optimizer, loss_fn, scaler, writer, step):
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

        # Tensorboard
        writer.add_scalar("Train Loss", loss, global_step=step)
        # writer.add_scalar("Train Loss", loss, global_step=step)
        step += 1


def transformation(image_height, image_width):
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
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

    return train_transform, val_transform


def main():
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=hyperparameters.BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=hyperparameters.NUM_EPOCHS,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=hyperparameters.LEARNING_RATE,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=hyperparameters.IMAGE_HEIGHT,
        help="Height of input images",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=hyperparameters.IMAGE_WIDTH,
        help="Width of input images",
    )
    parser.add_argument(
        "--load_model",
        action="store_true",
        help="Option to load pretrained model (default is False)",
    )

    args = parser.parse_args()

    print(f"Batch size: {args.batch_size}\nNumber of epochs: {args.num_epochs}\n")

    ################################################################################
    train_transform, val_transform = transformation(args.image_height, args.image_width)

    model = UNet(in_channels=3, out_channels=1).to(device=DEVICE)

    # Changing out_channels to ex. 3 & loss to cross entropy will yield class segmentation

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader, val_loader = get_loaders(
        hyperparameters.TRAIN_IMG_DIR,
        hyperparameters.TRAIN_MASK_DIR,
        hyperparameters.VAL_IMG_DIR,
        hyperparameters.VAL_MASK_DIR,
        args.batch_size,
        train_transform,
        val_transform,
        hyperparameters.NUM_WORKERS,
        hyperparameters.PIN_MEMORY,
    )

    if args.load_model:
        load_checkpoint(
            torch.load(
                os.path.join(
                    os.path.dirname(__file__), "checkpoints", "best_checkpoint.pth.tar"
                )
            ),
            model,
        )

    # check accuracy before training - in case of pretrained model we will see its performance
    check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), "runs"))

    step = 0
    best_dice_score = 0
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print("-" * 30)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, writer, step)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"epoch_{epoch}_checkpoint.pth.tar")

        # check accuracy
        dice_score = check_accuracy(val_loader, model, device=DEVICE)

        if dice_score > best_dice_score:
            best_dice_score = dice_score
            save_checkpoint(checkpoint, filename=f"best_checkpoint.pth.tar")

        # save examples to the folder
        if epoch % 5 == 0:
            save_predictions_as_imgs(
                val_loader,
                model,
                epoch,
                folder=os.path.join(os.path.dirname(__file__), "predictions"),
                device=DEVICE,
            )


if __name__ == "__main__":
    main()
