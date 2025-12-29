"""
Modern model definitions using transfer learning (MobileNetV2).
"""
from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
from keras.applications import MobileNetV2
from keras.optimizers import Adam

def create_transfer_model(num_classes=43, input_shape=(96,96,3), learning_rate=1e-3, fine_tune_at=None):
    """Create a MobileNetV2-based transfer learning model.

    Args:
        num_classes: number of output classes
        input_shape: input image shape
        learning_rate: optimizer learning rate
        fine_tune_at: layer index to start fine-tuning (None to keep base frozen)

    Returns:
        model: compiled Keras Model
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    if fine_tune_at is not None:
        # Unfreeze from fine_tune_at onward
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
