import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import argparse
import os

from model import UNet
from utils import load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Background removal")
    parser.add_argument(
        "--img_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "29bb3ece3180_11.jpg"),
        help="Path to the image to segment",
    )
    args = parser.parse_args()

    # Initialize the model
    model = UNet(in_channels=3, out_channels=1)
    load_checkpoint(
        checkpoint=torch.load(
            os.path.join(
                os.path.dirname(__file__), "checkpoints", "best_checkpoint.pth.tar"
            )
        ),
        model=model,
    )
    model.to(device=DEVICE)
    model.eval()

    # Load image
    img = Image.open(args.img_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 256)
            ),  # Resize the image to the input size of your model
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(img).unsqueeze(0).to(device=DEVICE)

    # Run inference
    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor))

    # Apply thresholding for binary segmentation
    predicted_mask = (output > 0.5).float()

    # Save the predicted mask
    utils.save_image(predicted_mask, f"segmented.png")

    # Display the saved mask
    predicted_mask_image = Image.open("predicted_mask.png")
    predicted_mask_image.show()


if __name__ == "__main__":
    main()
