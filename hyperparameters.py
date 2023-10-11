import os

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 8
NUM_WORKERS = 2  # parallel processes
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True  # faster data transfer from CPU to GPU
LOAD_MODEL = True  # True
TRAIN_IMG_DIR = os.path.join(os.path.dirname(__file__), "data", "train_images")
TRAIN_MASK_DIR = os.path.join(os.path.dirname(__file__), "data", "train_masks")
VAL_IMG_DIR = os.path.join(os.path.dirname(__file__), "data", "val_images")
VAL_MASK_DIR = os.path.join(os.path.dirname(__file__), "data", "val_masks")
