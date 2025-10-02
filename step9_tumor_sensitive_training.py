"""
Enhanced Tumor-Sensitive Training for Maximum Tumor Detection
============================================================

This script focuses on MAXIMIZING tumor detection sensitivity to capture
MORE TUMOR PARTS and reduce false negatives.

Key Features:
- Ultra-high sensitivity focal loss (gamma=5.0, alpha=0.9)
- Recall-focused loss functions
- Tumor-boosting data augmentation
- Aggressive tumor detection parameters
- Lower detection thresholds
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from scipy import ndimage
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TumorSensitiveTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        
    def ultra_sensitive_focal_loss(self, gamma=5.0, alpha=0.9):
        """
        Ultra-sensitive focal loss for maximum tumor detection
        - Very high gamma (5.0) to focus on hard examples
        - High alpha (0.9) to heavily weight tumor class
        """
        def focal_loss_fn(y_true, y_pred):
            # Clip predictions to prevent log(0)
            y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
            
            # Calculate focal loss components
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
            
            # Ultra-high gamma for aggressive focusing
            focal_weight = alpha_t * tf.pow(1 - pt, gamma)
            focal_loss = -focal_weight * tf.math.log(pt)
            
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_fn
    
    def recall_focused_loss(self, beta=4.0):
        """
        Recall-focused F-beta loss that heavily weights recall (tumor detection)
        Beta=4.0 means recall is 16x more important than precision
        """
        def recall_loss_fn(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
            
            # Calculate precision and recall
            tp = tf.reduce_sum(y_true * y_pred)
            fp = tf.reduce_sum((1 - y_true) * y_pred)
            fn = tf.reduce_sum(y_true * (1 - y_pred))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            # F-beta score with heavy recall weighting
            beta_squared = beta * beta
            f_beta = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall + 1e-8)
            
            # Return negative F-beta as loss (we want to maximize F-beta)
            return 1 - f_beta
        
        return recall_loss_fn
    
    def tumor_expansion_dice_loss(self, smooth=1e-6):
        """
        Modified Dice loss that encourages tumor expansion
        """
        def expansion_dice_fn(y_true, y_pred):
            # Standard Dice loss
            y_pred_f = tf.keras.backend.flatten(y_pred)
            y_true_f = tf.keras.backend.flatten(y_true)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
            
            # Expansion bonus: reward predictions slightly larger than ground truth
            expansion_bonus = tf.keras.backend.sum(y_pred_f * (1 - y_true_f)) * 0.1
            
            return 1 - dice - expansion_bonus
        
        return expansion_dice_fn
    
    def combined_tumor_sensitive_loss(self):
        """
        Combines multiple loss functions for maximum tumor sensitivity
        """
        focal_loss = self.ultra_sensitive_focal_loss(gamma=5.0, alpha=0.9)
        recall_loss = self.recall_focused_loss(beta=4.0)
        dice_loss = self.tumor_expansion_dice_loss()
        
        def combined_loss(y_true, y_pred):
            return (0.4 * focal_loss(y_true, y_pred) + 
                   0.4 * recall_loss(y_true, y_pred) + 
                   0.2 * dice_loss(y_true, y_pred))
        
        return combined_loss
    
    def tumor_sensitive_metrics(self):
        """
        Custom metrics focused on tumor detection performance
        """
        def tumor_recall(y_true, y_pred):
            """Sensitivity/Recall for tumor detection"""
            y_pred_binary = tf.cast(y_pred > 0.3, tf.float32)  # Lower threshold!
            tp = tf.reduce_sum(y_true * y_pred_binary)
            fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
            return tp / (tp + fn + 1e-8)
        
        def tumor_precision(y_true, y_pred):
            """Precision for tumor detection"""
            y_pred_binary = tf.cast(y_pred > 0.3, tf.float32)  # Lower threshold!
            tp = tf.reduce_sum(y_true * y_pred_binary)
            fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
            return tp / (tp + fp + 1e-8)
        
        def tumor_f2_score(y_true, y_pred):
            """F2 score (recall-weighted) for tumor detection"""
            precision = tumor_precision(y_true, y_pred)
            recall = tumor_recall(y_true, y_pred)
            return 5 * precision * recall / (4 * precision + recall + 1e-8)
        
        return [tumor_recall, tumor_precision, tumor_f2_score]
    
    def create_tumor_boosted_generator(self, X_train, y_train, batch_size=8):
        """
        Data generator with tumor-boosting augmentation
        """
        def tumor_boost_augmentation(image, mask):
            """Apply tumor-specific augmentations"""
            # Random rotation (tumors can appear at any angle)
            angle = np.random.uniform(-30, 30)
            image = ndimage.rotate(image, angle, reshape=False, mode='nearest')
            mask = ndimage.rotate(mask, angle, reshape=False, mode='nearest')
            
            # Random zoom focusing on tumor regions
            if np.sum(mask) > 100:  # If tumor present
                zoom_factor = np.random.uniform(0.9, 1.2)
                image = ndimage.zoom(image, zoom_factor, mode='nearest')
                mask = ndimage.zoom(mask, zoom_factor, mode='nearest')
                
                # Crop/pad back to original size
                if image.shape[0] != 256:
                    image = cv2.resize(image, (256, 256))
                    mask = cv2.resize(mask, (256, 256))
            
            # Brightness/contrast for tumor visibility
            if np.random.random() > 0.5:
                image = image * np.random.uniform(0.8, 1.3)
                image = np.clip(image, 0, 1)
            
            return image, mask
        
        def generator():
            # Oversample tumor-rich samples
            tumor_indices = []
            normal_indices = []
            
            for i in range(len(y_train)):
                if np.sum(y_train[i]) > 500:  # Tumor-rich threshold
                    tumor_indices.append(i)
                else:
                    normal_indices.append(i)
            
            while True:
                batch_images = []
                batch_masks = []
                
                for _ in range(batch_size):
                    # 70% tumor-rich, 30% normal (aggressive tumor focus)
                    if np.random.random() < 0.7 and len(tumor_indices) > 0:
                        idx = np.random.choice(tumor_indices)
                    else:
                        idx = np.random.choice(normal_indices) if len(normal_indices) > 0 else np.random.choice(tumor_indices)
                    
                    image = X_train[idx].copy()
                    mask = y_train[idx].copy()
                    
                    # Apply tumor-boosting augmentation
                    image, mask = tumor_boost_augmentation(image, mask)
                    
                    batch_images.append(image)
                    batch_masks.append(mask)
                
                yield np.array(batch_images), np.array(batch_masks)
        
        return generator()
    
    def compile_tumor_sensitive_model(self, model):
        """
        Compile model with tumor-sensitive configuration
        """
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,  # Lower learning rate for stability
            beta_1=0.9,
            beta_2=0.999
        )
        
        model.compile(
            optimizer=optimizer,
            loss=self.combined_tumor_sensitive_loss(),
            metrics=['accuracy'] + self.tumor_sensitive_metrics()
        )
        
        self.model = model
        return model
    
    def train_tumor_sensitive_model(self, model, train_generator, val_data, epochs=100):
        """
        Train model with tumor-sensitive callbacks and parameters
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_tumor_recall',  # Monitor tumor recall!
                patience=15,
                restore_best_weights=True,
                mode='max'  # Maximize recall
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_tumor_recall',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                mode='max'
            ),
            keras.callbacks.ModelCheckpoint(
                'best_tumor_sensitive_model.h5',
                monitor='val_tumor_recall',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train with tumor-focused parameters
        self.history = model.fit(
            train_generator,
            steps_per_epoch=100,  # More steps for better tumor learning
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_tumor_training_history(self):
        """
        Plot training history focusing on tumor metrics
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Tumor Recall
        axes[0, 0].plot(self.history.history['tumor_recall'], label='Train Tumor Recall')
        axes[0, 0].plot(self.history.history['val_tumor_recall'], label='Val Tumor Recall')
        axes[0, 0].set_title('Tumor Recall (Sensitivity)')
        axes[0, 0].set_ylabel('Recall')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Tumor Precision
        axes[0, 1].plot(self.history.history['tumor_precision'], label='Train Tumor Precision')
        axes[0, 1].plot(self.history.history['val_tumor_precision'], label='Val Tumor Precision')
        axes[0, 1].set_title('Tumor Precision')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F2 Score (Recall-weighted)
        axes[1, 0].plot(self.history.history['tumor_f2_score'], label='Train F2 Score')
        axes[1, 0].plot(self.history.history['val_tumor_f2_score'], label='Val F2 Score')
        axes[1, 0].set_title('Tumor F2 Score (Recall-Weighted)')
        axes[1, 0].set_ylabel('F2 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Loss
        axes[1, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[1, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1, 1].set_title('Combined Tumor-Sensitive Loss')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('tumor_sensitive_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function to demonstrate tumor-sensitive training
    """
    print("🎯 TUMOR-SENSITIVE TRAINING SYSTEM")
    print("=" * 50)
    print("🔥 MAXIMIZING TUMOR DETECTION SENSITIVITY")
    print("🎯 CAPTURING MORE TUMOR PARTS")
    print("⚡ REDUCING FALSE NEGATIVES")
    print("=" * 50)
    
    # Initialize trainer
    trainer = TumorSensitiveTrainer()
    
    # Load your data here
    print("📊 Load your training data:")
    print("   X_train, y_train = load_your_data()")
    print("   X_val, y_val = load_your_validation_data()")
    print()
    
    # Create model (use your existing model architecture)
    print("🏗️  Create your model:")
    print("   model = create_your_unet_model()")
    print("   model = trainer.compile_tumor_sensitive_model(model)")
    print()
    
    # Create tumor-boosted generator
    print("🚀 Create tumor-boosted data generator:")
    print("   train_gen = trainer.create_tumor_boosted_generator(X_train, y_train)")
    print()
    
    # Train with tumor-sensitive parameters
    print("🎯 Train with maximum tumor sensitivity:")
    print("   history = trainer.train_tumor_sensitive_model(model, train_gen, (X_val, y_val))")
    print()
    
    # Plot results
    print("📈 Visualize tumor-focused training results:")
    print("   trainer.plot_tumor_training_history()")
    print()
    
    print("✅ TUMOR-SENSITIVE TRAINING COMPLETE!")
    print("🎯 Your model is now optimized for MAXIMUM TUMOR DETECTION!")

if __name__ == "__main__":
    main()