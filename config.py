import os
import torch

# Device settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
KWARGS = {'pin_memory': True, 'num_workers': 4} if DEVICE == 'cude' else {}

# Data paths
IMAGE_PATH = 'images'
MASK_PATH = 'annotations/trimaps'

# Data attributes
IMAGE_SIZE = (256, 256)
N_CLASSES = 3
BATCH_SIZE = 8

# Model parameters
ENC_CHANNELS = (3, 64, 128, 256, 512)
DEC_CHANNELS = (512, 256, 128, 64)

# Train / test split rate
SPLIT_RATE = 0.2

# Learning parameters
LEARNING_RATE = 0.0001
EPOCHS = 10

# Output paths
OUTPUT_PATH = 'output'
TEST_IMGS_PATH = os.path.join(OUTPUT_PATH, 'test_imgs.pickle')
TEST_MASKS_PATH = os.path.join(OUTPUT_PATH, 'test_masks.pickle')
MODEL_PATH = os.path.join(OUTPUT_PATH, 'unet_model.pth')
HISTORY_PATH = os.path.join(OUTPUT_PATH, 'unet_history.pickle')
HISTORY_PLOT_PATH = os.path.join(OUTPUT_PATH, 'history.png')
PRED_PLOT_PATH = os.path.join(OUTPUT_PATH, 'pred.png')
