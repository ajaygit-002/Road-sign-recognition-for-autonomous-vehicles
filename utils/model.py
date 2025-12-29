"""
Model architecture utilities for the road sign recognition project.
Defines the CNN model structure and provides functions for model management.
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from .config import IMG_SIZE, USE_TRANSFER_LEARNING, INPUT_SHAPE, LEARNING_RATE, FINE_TUNE_AT

if USE_TRANSFER_LEARNING:
    try:
        from .modern_model import create_transfer_model
    except Exception:
        create_transfer_model = None


def create_model(num_classes=43, input_shape=None):
    """Create a model. If transfer learning is enabled, return a pretrained backbone model.

    Args:
        num_classes: Number of output classes
        input_shape: Optional input shape override

    Returns:
        model: Uncompiled/compiled Keras model depending on branch
    """
    if input_shape is None:
        input_shape = INPUT_SHAPE

    if USE_TRANSFER_LEARNING and create_transfer_model is not None:
        return create_transfer_model(num_classes=num_classes, input_shape=input_shape,
                                     learning_rate=LEARNING_RATE, fine_tune_at=FINE_TUNE_AT)

    model = Sequential()

    # First convolutional block
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', 
                     input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    
    # Second convolutional block
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def compile_model(model, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
    """
    Compile the model with specified loss and optimizer.
    
    Args:
        model: Keras model to compile
        loss: Loss function (default: 'categorical_crossentropy')
        optimizer: Optimizer (default: 'adam')
        metrics: Metrics to track (default: ['accuracy'])
    
    Returns:
        model: Compiled model
    """
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
