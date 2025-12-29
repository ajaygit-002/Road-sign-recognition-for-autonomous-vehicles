"""
Configuration constants for the project.
"""
VERSION = "2.0"

# Image and training parameters
# Default to a larger size when using modern backbones (MobileNetV2)
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 25

# Transfer learning / modern algorithm options
USE_TRANSFER_LEARNING = True
BASE_MODEL_NAME = 'MobileNetV2'  # currently supported: MobileNetV2
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
FINE_TUNE_AT = 100  # layer from which to fine-tune (None to skip)

# Enable simple training-time augmentation
AUGMENTATION = True
AUGMENTATION_PARAMS = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': False,
    'brightness_range': (0.8, 1.2)
}

# Optimizer / scheduler choices can be expanded later
LEARNING_RATE = 1e-3
