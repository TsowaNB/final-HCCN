"""
Step 4: Training Pipeline
Hybrid U-Net Model Training for Lower-grade Glioma Segmentation
"""

import os

# Configure TensorFlow for GPU P100 training
import tensorflow as tf

# GPU Configuration for P100
print(" Configuring TensorFlow for GPU P100 training...")

# Enable GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" Found {len(gpus)} GPU(s) - Memory growth enabled")
    except RuntimeError as e:
        print(f" GPU configuration error: {e}")
else:
    print(" No GPU detected - falling back to CPU")

# Now import other TF modules
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, BinaryIoU

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for segmentation evaluation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function."""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combined binary crossentropy and dice loss."""
    bce = BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

def create_callbacks(model_name, patience=10):
    """Create training callbacks."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks

def compile_model(model, learning_rate=1e-4):
    """Compile model with optimizer, loss, and metrics."""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=[dice_coefficient, BinaryAccuracy(), Precision(), Recall(), BinaryIoU()]
    )
    return model

def train_model(model, train_gen, val_gen, model_name, epochs=30):
    """Train model with generators and callbacks."""
    print(f" Training {model_name}...")
    
    callbacks = create_callbacks(model_name)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f" {model_name} training completed!")
    return history

def main():
    """Main training pipeline optimized for GPU P100."""
    
    # Load and prepare data
    print(" Preparing dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(DATASET_PATH)
    
    # Create data generators with GPU-optimized batch size
    batch_size = 16  # Larger batch size for GPU P100 (16GB memory)
    train_gen = MRIDataGenerator(X_train, y_train, batch_size=batch_size, augment=True)
    val_gen = MRIDataGenerator(X_val, y_val, batch_size=batch_size, augment=False)
    
    print(f" Using batch size: {batch_size} (optimized for GPU P100)")

        # Train Hybrid Quantum U-Net
    print("\n Training Hybrid Quantum U-Net...")
    hybrid_model = create_hybrid_quantum_unet()
    hybrid_model = compile_model(hybrid_model, learning_rate=1e-3)  # Higher LR for GPU
    hybrid_history = train_model(hybrid_model, train_gen, val_gen, "hybrid_quantum_unet", epochs=30)
    
    
    # Train Classical U-Net
    print("\n Training Classical U-Net...")
    classical_model = create_classical_unet()
    classical_model = compile_model(classical_model, learning_rate=1e-3)  # Higher LR for GPU
    classical_history = train_model(classical_model, train_gen, val_gen, "classical_unet", epochs=30)

    print("\n Training pipeline completed successfully!")
    print(" GPU P100 training provides faster convergence and higher throughput!")
    return hybrid_history, classical_history

if __name__ == "__main__":
    main()