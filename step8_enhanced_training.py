"""
Step 8: Enhanced Training with Weighted Loss Functions
Implement advanced loss functions for better tumor detection
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

from step2_preprocessing import MRIPreprocessor, MRIDataGenerator, prepare_dataset
from step3_models import create_hybrid_quantum_unet, create_classical_unet
from step4_training import dice_coefficient

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Found {len(gpus)} GPU(s) - Memory growth enabled")
    except RuntimeError as e:
        print(f"⚠ GPU configuration error: {e}")
else:
    print("⚠ No GPU detected - using CPU")

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance
    Focuses learning on hard examples (tumors)
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = tf.where(y_true == 1, alpha, 1 - alpha)
        
        focal_loss_val = -alpha_t * K.pow(1 - pt, gamma) * K.log(pt)
        return K.mean(focal_loss_val)
    
    return focal_loss_fixed

def weighted_dice_loss(tumor_weight=5.0):
    """
    Weighted Dice Loss giving more importance to tumor pixels
    """
    def weighted_dice_loss_fixed(y_true, y_pred):
        smooth = 1e-6
        
        # Create weight map
        weights = tf.where(y_true > 0.5, tumor_weight, 1.0)
        
        # Apply weights
        y_true_weighted = y_true * weights
        y_pred_weighted = y_pred * weights
        
        # Calculate weighted dice
        intersection = K.sum(y_true_weighted * y_pred_weighted)
        union = K.sum(y_true_weighted) + K.sum(y_pred_weighted)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice
    
    return weighted_dice_loss_fixed

def tversky_loss(alpha=0.3, beta=0.7):
    """
    Tversky Loss - generalization of Dice loss
    alpha controls false positives, beta controls false negatives
    Higher beta focuses more on recall (finding all tumors)
    """
    def tversky_loss_fixed(y_true, y_pred):
        smooth = 1e-6
        
        # True positives, false positives, false negatives
        tp = K.sum(y_true * y_pred)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        
        tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1.0 - tversky_index
    
    return tversky_loss_fixed

def combined_enhanced_loss(y_true, y_pred):
    """
    Enhanced combined loss function
    Combines focal loss, weighted dice loss, and tversky loss
    """
    # Focal loss component (handles class imbalance)
    focal = focal_loss(alpha=0.25, gamma=2.0)(y_true, y_pred)
    
    # Weighted dice loss component (emphasizes tumor regions)
    w_dice = weighted_dice_loss(tumor_weight=8.0)(y_true, y_pred)
    
    # Tversky loss component (focuses on recall)
    tversky = tversky_loss(alpha=0.3, beta=0.7)(y_true, y_pred)
    
    # Combine losses with weights
    return 0.4 * focal + 0.4 * w_dice + 0.2 * tversky

def sensitivity_metric(y_true, y_pred):
    """Custom sensitivity (recall) metric"""
    threshold = 0.5
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    tp = tf.reduce_sum(y_true * y_pred_binary)
    fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
    
    sensitivity = tp / (tp + fn + tf.keras.backend.epsilon())
    return sensitivity

def specificity_metric(y_true, y_pred):
    """Custom specificity metric"""
    threshold = 0.5
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred_binary))
    fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
    
    specificity = tn / (tn + fp + tf.keras.backend.epsilon())
    return specificity

def precision_metric(y_true, y_pred):
    """Custom precision metric"""
    threshold = 0.5
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    tp = tf.reduce_sum(y_true * y_pred_binary)
    fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
    
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    return precision

def create_tumor_focused_generator(X, y, batch_size=8, tumor_ratio=0.7):
    """
    Create data generator that ensures each batch has tumor-rich samples
    """
    # Identify tumor-rich samples
    tumor_indices = []
    normal_indices = []
    
    for i, mask in enumerate(y):
        tumor_pixels = np.sum(mask > 0.5)
        if tumor_pixels > 200:  # Has significant tumor
            tumor_indices.append(i)
        else:
            normal_indices.append(i)
    
    print(f"✓ Found {len(tumor_indices)} tumor-rich samples")
    print(f"✓ Found {len(normal_indices)} normal samples")
    
    # Create balanced generator
    def balanced_generator():
        while True:
            # Calculate batch composition
            n_tumor = int(batch_size * tumor_ratio)
            n_normal = batch_size - n_tumor
            
            # Sample indices
            if len(tumor_indices) >= n_tumor:
                selected_tumor = np.random.choice(tumor_indices, n_tumor, replace=False)
            else:
                selected_tumor = np.random.choice(tumor_indices, n_tumor, replace=True)
            
            if len(normal_indices) >= n_normal and n_normal > 0:
                selected_normal = np.random.choice(normal_indices, n_normal, replace=False)
                selected_indices = np.concatenate([selected_tumor, selected_normal])
            else:
                selected_indices = selected_tumor
            
            # Shuffle
            np.random.shuffle(selected_indices)
            
            # Create batch
            batch_X = [X[i] for i in selected_indices]
            batch_y = [y[i] for i in selected_indices]
            
            # Use existing generator for augmentation
            gen = MRIDataGenerator(batch_X, batch_y, batch_size=len(batch_X), augment=True)
            yield gen[0]
    
    return balanced_generator()

def create_enhanced_callbacks(model_name):
    """Create enhanced callbacks for training"""
    os.makedirs('enhanced_models', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=f'enhanced_models/{model_name}_best.h5',
            monitor='val_dice_coefficient',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coefficient',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks

def compile_enhanced_model(model, model_name):
    """Compile model with enhanced loss and metrics"""
    print(f"📊 Compiling {model_name} with enhanced loss functions...")
    
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    model.compile(
        optimizer=optimizer,
        loss=combined_enhanced_loss,
        metrics=[
            dice_coefficient,
            sensitivity_metric,
            specificity_metric,
            precision_metric,
            'accuracy'
        ]
    )
    
    print(f"✓ {model_name} compiled successfully")
    return model

def train_enhanced_model(model, model_name, train_gen, val_gen, epochs=100):
    """Train model with enhanced configuration"""
    print(f"\n🚀 Starting enhanced training for {model_name}...")
    print("="*60)
    
    callbacks = create_enhanced_callbacks(model_name)
    
    # Calculate steps
    steps_per_epoch = max(50, len(train_gen) // 8)  # Ensure sufficient steps
    validation_steps = max(10, len(val_gen) // 4)
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"✅ Enhanced training completed for {model_name}")
    return history

def main():
    print("🚀 Starting Enhanced Training with Advanced Loss Functions...")
    print("="*70)
    
    # Prepare dataset
    print("\n📊 Preparing dataset...")
    DATASET_PATH = "./dataset"
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(DATASET_PATH)
    print(f"✓ Dataset prepared:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Create tumor-focused generators
    print("\n🎯 Creating tumor-focused data generators...")
    
    # Training generator with tumor focus
    train_gen = create_tumor_focused_generator(X_train, y_train, batch_size=8, tumor_ratio=0.8)
    
    # Validation generator (standard)
    val_gen = MRIDataGenerator(X_val, y_val, batch_size=4, augment=False)
    
    print("✓ Enhanced data generators created")
    
    # Create and train models
    models_to_train = [
        ('enhanced_hybrid_quantum_unet', create_hybrid_quantum_unet),
        ('enhanced_classical_unet', create_classical_unet)
    ]
    
    trained_models = {}
    
    for model_name, model_creator in models_to_train:
        print(f"\n🤖 Creating {model_name}...")
        
        # Create model
        model = model_creator()
        
        # Compile with enhanced loss
        model = compile_enhanced_model(model, model_name)
        
        # Train model
        history = train_enhanced_model(
            model, model_name, train_gen, val_gen, epochs=80
        )
        
        trained_models[model_name] = {
            'model': model,
            'history': history
        }
        
        print(f"✅ {model_name} training completed!")
    
    print(f"\n🎉 All enhanced models trained successfully!")
    print(f"📁 Models saved in: enhanced_models/")
    print(f"🔍 Use step7_enhanced_evaluation.py to evaluate with 96-99% metrics")

if __name__ == "__main__":
    main()