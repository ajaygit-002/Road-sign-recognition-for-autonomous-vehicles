"""
Training script for the road sign recognition model.
Run this script separately to train the model and save it to the models/ folder.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_dataset, prepare_training_data
from utils.model import create_model, compile_model
from utils.classes import NUM_CLASSES
from utils.config import BATCH_SIZE, EPOCHS, AUGMENTATION, AUGMENTATION_PARAMS, IMG_SIZE, INPUT_SHAPE, USE_TRANSFER_LEARNING, LEARNING_RATE, FINE_TUNE_AT


def main():
    """Main training function."""
    
    # Get paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(current_dir, 'dataset', 'train')
    models_dir = os.path.join(current_dir, 'models')
    
    print("=" * 60)
    print("Road Sign Recognition - Training Script")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"\nError: Dataset not found at {dataset_path}")
        print("Please ensure you have the dataset folder with 'images' and 'labels' subdirectories.")
        return
    
    print(f"\nDataset path: {dataset_path}")
    print(f"Models directory: {models_dir}")
    
    # Load dataset
    print("\n" + "-" * 60)
    print("Loading dataset...")
    print("-" * 60)
    data, labels = load_dataset(dataset_path, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    
    if len(data) == 0:
        print("Error: No data loaded. Exiting.")
        return
    
    # Prepare training data
    print("\n" + "-" * 60)
    print("Preparing training data...")
    print("-" * 60)
    X_train, X_test, y_train, y_test = prepare_training_data(
        data, labels, test_size=0.2, num_classes=NUM_CLASSES
    )
    
    if X_train is None:
        print("Error: Failed to prepare data. Exiting.")
        return
    
    # Create and compile model
    print("\n" + "-" * 60)
    print("Creating and compiling model...")
    print("-" * 60)
    model = create_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
    # create_model may already return a compiled transfer model; only compile if uncompiled
    try:
        # If model has attribute 'optimizer' set, assume compiled
        _ = model.optimizer
    except Exception:
        model = compile_model(model)
    print("Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("\n" + "-" * 60)
    print("Training model...")
    print("-" * 60)
    if AUGMENTATION:
        print("Using data augmentation for training.")
        datagen = ImageDataGenerator(**AUGMENTATION_PARAMS)
        datagen.fit(X_train)
        steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
    
    # Save model
    print("\n" + "-" * 60)
    print("Saving model...")
    print("-" * 60)
    model_path = os.path.join(models_dir, 'traffic_sign_model.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Create outputs directory for graphs
    outputs_dir = os.path.join(current_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Plot and save accuracy graph
    print("\nGenerating accuracy graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    accuracy_path = os.path.join(outputs_dir, 'accuracy.png')
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy graph saved to: {accuracy_path}")
    
    # Plot and save loss graph
    print("Generating loss graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(outputs_dir, 'loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss graph saved to: {loss_path}")
    
    # Print final metrics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
