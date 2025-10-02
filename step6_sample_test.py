"""
Step 6: Sample Test
Test specific sample (590) to get precision, accuracy, F1-score, and recall
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model

from step2_preprocessing import MRIPreprocessor, MRIDataGenerator, prepare_dataset
from step3_models import create_hybrid_quantum_unet, create_classical_unet, QuantumLayer
from step4_training import dice_coefficient, dice_loss, combined_loss

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

def load_trained_models():
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'combined_loss': combined_loss,
        'QuantumLayer': QuantumLayer
    }
    
    models = {}
    
    hybrid_path = 'models/hybrid_quantum_unet_best.h5'
    if os.path.exists(hybrid_path):
        try:
            models['Hybrid Quantum U-Net'] = load_model(hybrid_path, custom_objects=custom_objects)
            print("✓ Hybrid Quantum U-Net loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading Hybrid Quantum U-Net: {e}")
            models['Hybrid Quantum U-Net'] = create_hybrid_quantum_unet()
    else:
        print("⚠ Hybrid model not found, creating new model...")
        models['Hybrid Quantum U-Net'] = create_hybrid_quantum_unet()
    
    classical_path = 'models/classical_unet_best.h5'
    if os.path.exists(classical_path):
        try:
            models['Classical U-Net'] = load_model(classical_path, custom_objects=custom_objects)
            print("✓ Classical U-Net loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading Classical U-Net: {e}")
            models['Classical U-Net'] = create_classical_unet()
    else:
        print("⚠ Classical model not found, creating new model...")
        models['Classical U-Net'] = create_classical_unet()
    
    return models

def test_sample_590(models, X_test, y_test, sample_index=590):
    print(f"\n🔍 Testing Sample {sample_index}")
    print("="*50)
    
    if sample_index >= len(X_test):
        print(f"⚠ Sample {sample_index} not found. Dataset has {len(X_test)} samples.")
        sample_index = min(sample_index, len(X_test) - 1)
        print(f"Using sample {sample_index} instead.")
    
    # Create single sample generator
    single_sample_gen = MRIDataGenerator([X_test[sample_index]], [y_test[sample_index]], 
                                       batch_size=1, augment=False)
    
    # Get the sample
    sample_x, sample_y = single_sample_gen[0]
    ground_truth = sample_y[0].squeeze()
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n📊 Testing {model_name} on Sample {sample_index}")
        print("-" * 40)
        
        # Get prediction
        prediction = model.predict(sample_x, verbose=0)[0].squeeze()
        
        # Convert to binary
        y_true_binary = (ground_truth.flatten() > 0.5).astype(int)
        y_pred_binary = (prediction.flatten() > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        
        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'prediction': prediction,
            'ground_truth': ground_truth
        }
        
        # Print metrics
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        report = classification_report(y_true_binary, y_pred_binary, 
                                     target_names=['Background', 'Tumor'], 
                                     digits=4, zero_division=0)
        print(report)
    
    # Visualize results
    visualize_sample_results(sample_x[0], results, sample_index)
    
    return results

def visualize_sample_results(original_image, results, sample_index):
    print(f"\n📈 Generating visualization for Sample {sample_index}...")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_image.squeeze(), cmap='gray')
    axes[0].set_title(f'Original MRI - Sample {sample_index}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth
    ground_truth = list(results.values())[0]['ground_truth']
    axes[1].imshow(ground_truth, cmap='Reds', alpha=0.7)
    axes[1].imshow(original_image.squeeze(), cmap='gray', alpha=0.3)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Model predictions
    model_names = list(results.keys())
    colors = ['Blues', 'Greens']
    
    for i, model_name in enumerate(model_names[:2]):
        prediction = results[model_name]['prediction']
        axes[i+2].imshow(prediction, cmap=colors[i], alpha=0.7)
        axes[i+2].imshow(original_image.squeeze(), cmap='gray', alpha=0.3)
        axes[i+2].set_title(f'{model_name}\nF1: {results[model_name]["f1_score"]:.4f}', 
                           fontsize=12, fontweight='bold')
        axes[i+2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'sample_{sample_index}_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_models_on_sample(results, sample_index):
    print(f"\n📊 Model Comparison for Sample {sample_index}")
    print("="*60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    
    for model_name, model_results in results.items():
        print(f"{model_name:<20} {model_results['accuracy']:<10.4f} "
              f"{model_results['precision']:<10.4f} {model_results['recall']:<10.4f} "
              f"{model_results['f1_score']:<10.4f}")
    
    # Find best performing model
    best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
    print(f"\n🏆 Best performing model: {best_model} (F1-Score: {results[best_model]['f1_score']:.4f})")

def main():
    print("🚀 Starting Sample 590 Test...")
    print("="*50)
    
    # Prepare dataset
    print("\n📊 Preparing dataset...")
    DATASET_PATH = "./dataset"
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(DATASET_PATH)
    print(f"✓ Dataset prepared: {len(X_test)} test samples")
    
    # Load models
    print("\n🤖 Loading trained models...")
    models = load_trained_models()
    
    # Test sample 590
    results = test_sample_590(models, X_test, y_test, sample_index=590)
    
    # Compare models
    compare_models_on_sample(results, 590)
    
    print(f"\n✅ Sample 590 testing completed!")
    print(f"📁 Visualization saved as: sample_590_results.png")

if __name__ == "__main__":
    main()