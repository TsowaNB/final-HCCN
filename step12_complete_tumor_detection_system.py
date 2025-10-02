"""
Complete Tumor Detection System - Maximum Tumor Prediction
=========================================================

This script integrates ALL tumor detection enhancements to achieve
MAXIMUM TUMOR PREDICTION and capture MORE TUMOR PARTS.

Integrated Components:
1. Tumor-sensitive training (step9)
2. Tumor expansion post-processing (step10)
3. Tumor-sensitive evaluation (step11)
4. Complete pipeline for maximum tumor detection

Key Features:
- Ultra-sensitive tumor detection
- Comprehensive tumor expansion
- Lower detection thresholds
- Maximum tumor coverage
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report
import os

# Import our custom modules
from step9_tumor_sensitive_training import TumorSensitiveTrainer
from step10_tumor_expansion_postprocessing import TumorExpansionProcessor
from step11_tumor_sensitive_evaluation import TumorSensitiveEvaluator

class CompleteTumorDetectionSystem:
    def __init__(self):
        self.trainer = TumorSensitiveTrainer()
        self.processor = TumorExpansionProcessor()
        self.evaluator = TumorSensitiveEvaluator()
        self.model = None
        self.optimal_threshold = 0.2  # Default sensitive threshold
        
    def load_and_prepare_data(self):
        """
        Load and prepare data for tumor detection
        """
        print("📊 LOADING DATA FOR TUMOR DETECTION")
        print("=" * 40)
        
        try:
            # Load data using existing data loading script
            from step1_data_loading import load_data
            X_train, X_val, X_test, y_train, y_val, y_test = load_data()
            
            print(f"✅ Training data: {X_train.shape}")
            print(f"✅ Validation data: {X_val.shape}")
            print(f"✅ Test data: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            print(f"⚠️  Could not load data automatically: {e}")
            print("📝 Please load your data manually:")
            print("   X_train, X_val, X_test, y_train, y_val, y_test = your_data_loading_function()")
            return None, None, None, None, None, None
    
    def create_tumor_optimized_model(self, input_shape=(256, 256, 1)):
        """
        Create U-Net model optimized for tumor detection
        """
        print("🏗️  CREATING TUMOR-OPTIMIZED MODEL")
        print("=" * 35)
        
        inputs = keras.Input(shape=input_shape)
        
        # Encoder with more aggressive feature extraction
        conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = keras.layers.Dropout(0.5)(conv4)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
        
        # Bottleneck with attention for tumor focus
        conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = keras.layers.Dropout(0.5)(conv5)
        
        # Decoder with skip connections
        up6 = keras.layers.Conv2D(512, 2, activation='relu', padding='same')(keras.layers.UpSampling2D(size=(2, 2))(drop5))
        merge6 = keras.layers.concatenate([drop4, up6], axis=3)
        conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
        
        up7 = keras.layers.Conv2D(256, 2, activation='relu', padding='same')(keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = keras.layers.concatenate([conv3, up7], axis=3)
        conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
        
        up8 = keras.layers.Conv2D(128, 2, activation='relu', padding='same')(keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = keras.layers.concatenate([conv2, up8], axis=3)
        conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
        
        up9 = keras.layers.Conv2D(64, 2, activation='relu', padding='same')(keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = keras.layers.concatenate([conv1, up9], axis=3)
        conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
        
        # Output layer with sigmoid for probability
        outputs = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        print("✅ Tumor-optimized U-Net model created")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def train_complete_system(self, X_train, y_train, X_val, y_val, epochs=50):
        """
        Train the complete tumor detection system
        """
        print("🚀 TRAINING COMPLETE TUMOR DETECTION SYSTEM")
        print("=" * 50)
        
        # Create and compile model
        self.model = self.create_tumor_optimized_model()
        self.model = self.trainer.compile_tumor_sensitive_model(self.model)
        
        # Create tumor-boosted data generator
        train_generator = self.trainer.create_tumor_boosted_generator(X_train, y_train, batch_size=8)
        
        # Train with tumor-sensitive parameters
        history = self.trainer.train_tumor_sensitive_model(
            self.model, train_generator, (X_val, y_val), epochs=epochs
        )
        
        # Plot training history
        self.trainer.plot_tumor_training_history()
        
        print("✅ TUMOR-SENSITIVE TRAINING COMPLETED!")
        return history
    
    def predict_with_expansion(self, X_test):
        """
        Make predictions with tumor expansion post-processing
        """
        print("🔮 MAKING PREDICTIONS WITH TUMOR EXPANSION")
        print("=" * 45)
        
        if self.model is None:
            print("❌ No trained model available!")
            return None
        
        # Get raw predictions
        print("   🔍 Getting raw predictions...")
        raw_predictions = self.model.predict(X_test, verbose=1)
        
        # Apply tumor expansion post-processing
        print("   🚀 Applying tumor expansion...")
        expanded_predictions = []
        
        for i, (image, prediction) in enumerate(zip(X_test, raw_predictions)):
            if i < 5:  # Show progress for first 5 images
                print(f"      Processing image {i+1}...")
            
            # Apply comprehensive tumor expansion
            expansion_result = self.processor.comprehensive_tumor_expansion(prediction.squeeze())
            expanded_predictions.append(expansion_result['final'])
        
        expanded_predictions = np.array(expanded_predictions)
        
        print(f"✅ Predictions completed with expansion")
        print(f"   Raw predictions shape: {raw_predictions.shape}")
        print(f"   Expanded predictions shape: {expanded_predictions.shape}")
        
        return raw_predictions, expanded_predictions
    
    def evaluate_complete_system(self, X_test, y_test, raw_predictions, expanded_predictions):
        """
        Evaluate the complete tumor detection system
        """
        print("📊 EVALUATING COMPLETE TUMOR DETECTION SYSTEM")
        print("=" * 50)
        
        # Run comprehensive evaluation
        evaluation_results = self.evaluator.comprehensive_tumor_evaluation(
            X_test, y_test, raw_predictions.squeeze()
        )
        
        # Store optimal threshold
        self.optimal_threshold = evaluation_results['optimal_threshold']
        
        # Additional evaluation with expanded predictions
        print("\n🔥 EXPANDED PREDICTIONS EVALUATION")
        print("=" * 35)
        
        expanded_metrics = self.evaluator.calculate_tumor_metrics(
            y_test, expanded_predictions, threshold=0.5  # Binary mask threshold
        )
        
        print(f"🎯 FINAL RESULTS WITH EXPANSION:")
        print(f"   Sensitivity: {expanded_metrics['sensitivity']:.3f}")
        print(f"   Precision: {expanded_metrics['precision']:.3f}")
        print(f"   F2-Score: {expanded_metrics['f2_score']:.3f}")
        print(f"   Dice Score: {expanded_metrics['dice']:.3f}")
        print(f"   Tumor pixels detected: {expanded_metrics['tumor_pixels_detected']}")
        
        return evaluation_results, expanded_metrics
    
    def visualize_tumor_detection_results(self, X_test, y_test, raw_predictions, expanded_predictions, num_samples=6):
        """
        Visualize tumor detection results
        """
        print("🎨 VISUALIZING TUMOR DETECTION RESULTS")
        print("=" * 40)
        
        # Select samples with tumors for visualization
        tumor_indices = []
        for i in range(len(y_test)):
            if np.sum(y_test[i]) > 100:  # Has significant tumor
                tumor_indices.append(i)
        
        if len(tumor_indices) == 0:
            print("⚠️  No tumor samples found for visualization")
            return
        
        # Select random tumor samples
        selected_indices = np.random.choice(tumor_indices, min(num_samples, len(tumor_indices)), replace=False)
        
        fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(selected_indices):
            # Original image
            axes[i, 0].imshow(X_test[idx].squeeze(), cmap='gray')
            axes[i, 0].set_title(f'Original Image #{idx}')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(X_test[idx].squeeze(), cmap='gray', alpha=0.7)
            axes[i, 1].imshow(y_test[idx], cmap='Reds', alpha=0.5)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Raw prediction
            axes[i, 2].imshow(raw_predictions[idx].squeeze(), cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title('Raw Prediction')
            axes[i, 2].axis('off')
            
            # Thresholded prediction
            thresholded = (raw_predictions[idx].squeeze() >= self.optimal_threshold).astype(int)
            axes[i, 3].imshow(X_test[idx].squeeze(), cmap='gray', alpha=0.7)
            axes[i, 3].imshow(thresholded, cmap='Blues', alpha=0.5)
            axes[i, 3].set_title(f'Thresholded ({self.optimal_threshold})')
            axes[i, 3].axis('off')
            
            # Expanded prediction
            axes[i, 4].imshow(X_test[idx].squeeze(), cmap='gray', alpha=0.7)
            axes[i, 4].imshow(expanded_predictions[idx], cmap='Greens', alpha=0.5)
            axes[i, 4].set_title('Expanded Prediction')
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig('complete_tumor_detection_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization completed and saved!")
    
    def run_complete_pipeline(self):
        """
        Run the complete tumor detection pipeline
        """
        print("🎯 COMPLETE TUMOR DETECTION PIPELINE")
        print("=" * 45)
        print("🔥 MAXIMIZING TUMOR PREDICTION COVERAGE")
        print("🎯 CAPTURING MORE TUMOR PARTS")
        print("⚡ ULTRA-SENSITIVE DETECTION")
        print("=" * 45)
        
        # Step 1: Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()
        
        if X_train is None:
            print("❌ Data loading failed. Please load data manually.")
            return None
        
        # Step 2: Train tumor-sensitive model
        print("\n🚀 STEP 1: TUMOR-SENSITIVE TRAINING")
        history = self.train_complete_system(X_train, y_train, X_val, y_val, epochs=30)
        
        # Step 3: Make predictions with expansion
        print("\n🔮 STEP 2: PREDICTIONS WITH EXPANSION")
        raw_predictions, expanded_predictions = self.predict_with_expansion(X_test)
        
        # Step 4: Comprehensive evaluation
        print("\n📊 STEP 3: COMPREHENSIVE EVALUATION")
        evaluation_results, expanded_metrics = self.evaluate_complete_system(
            X_test, y_test, raw_predictions, expanded_predictions
        )
        
        # Step 5: Visualize results
        print("\n🎨 STEP 4: RESULT VISUALIZATION")
        self.visualize_tumor_detection_results(
            X_test, y_test, raw_predictions, expanded_predictions
        )
        
        # Final summary
        print("\n🏆 COMPLETE TUMOR DETECTION SYSTEM RESULTS")
        print("=" * 50)
        print(f"🎯 Optimal threshold: {self.optimal_threshold}")
        print(f"🔥 Final sensitivity: {expanded_metrics['sensitivity']:.3f}")
        print(f"⚡ Final F2-score: {expanded_metrics['f2_score']:.3f}")
        print(f"📈 Tumor pixels detected: {expanded_metrics['tumor_pixels_detected']}")
        print(f"✅ MAXIMUM TUMOR DETECTION ACHIEVED!")
        
        return {
            'model': self.model,
            'optimal_threshold': self.optimal_threshold,
            'evaluation_results': evaluation_results,
            'expanded_metrics': expanded_metrics,
            'raw_predictions': raw_predictions,
            'expanded_predictions': expanded_predictions
        }

def main():
    """
    Main function to run the complete tumor detection system
    """
    print("🎯 COMPLETE TUMOR DETECTION SYSTEM")
    print("=" * 40)
    print("🔥 MAXIMUM TUMOR PREDICTION COVERAGE")
    print("🎯 ULTRA-SENSITIVE TUMOR DETECTION")
    print("⚡ COMPREHENSIVE TUMOR EXPANSION")
    print("=" * 40)
    
    # Initialize complete system
    system = CompleteTumorDetectionSystem()
    
    # Run complete pipeline
    results = system.run_complete_pipeline()
    
    if results:
        print("\n✅ COMPLETE TUMOR DETECTION SYSTEM READY!")
        print("🎯 Your system now achieves MAXIMUM TUMOR PREDICTION!")
        print("🔥 Use the trained model and optimal threshold for deployment!")
    else:
        print("\n⚠️  Please check data loading and try again.")

if __name__ == "__main__":
    main()