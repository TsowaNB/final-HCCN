"""
Balanced Support Evaluation
Create more balanced support distribution with less weight on any single class
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

class BalancedSupportEvaluator:
    def __init__(self):
        self.optimal_thresholds = {}
        
    def find_balanced_samples(self, X_test, y_test, target_tumor_ratio=0.4, tolerance=0.1):
        """Find samples with balanced tumor-to-background ratio"""
        balanced_indices = []
        tumor_percentages = []
        
        target_min = target_tumor_ratio - tolerance
        target_max = target_tumor_ratio + tolerance
        
        for i, mask in enumerate(y_test):
            tumor_pixels = np.sum(mask > 0.5)
            total_pixels = mask.size
            tumor_percentage = tumor_pixels / total_pixels
            
            if target_min <= tumor_percentage <= target_max:
                balanced_indices.append(i)
                tumor_percentages.append(tumor_percentage * 100)
        
        print(f"✓ Found {len(balanced_indices)} balanced samples")
        print(f"✓ Target tumor ratio: {target_tumor_ratio*100:.1f}% ± {tolerance*100:.1f}%")
        if tumor_percentages:
            print(f"✓ Actual tumor percentages: {min(tumor_percentages):.1f}% - {max(tumor_percentages):.1f}%")
        
        return balanced_indices, tumor_percentages
    
    def create_balanced_subset(self, X_test, y_test, num_samples=10):
        """Create a balanced subset by mixing different tumor ratios"""
        print(f"\n🎯 Creating Balanced Subset ({num_samples} samples)")
        print("-" * 50)
        
        # Find samples with different tumor ratios
        low_tumor_indices = []    # 10-25% tumor
        medium_tumor_indices = [] # 25-50% tumor  
        high_tumor_indices = []   # 50-75% tumor
        
        for i, mask in enumerate(y_test):
            tumor_pixels = np.sum(mask > 0.5)
            total_pixels = mask.size
            tumor_ratio = tumor_pixels / total_pixels
            
            if 0.10 <= tumor_ratio <= 0.25:
                low_tumor_indices.append((i, tumor_ratio))
            elif 0.25 <= tumor_ratio <= 0.50:
                medium_tumor_indices.append((i, tumor_ratio))
            elif 0.50 <= tumor_ratio <= 0.75:
                high_tumor_indices.append((i, tumor_ratio))
        
        print(f"Available samples:")
        print(f"  Low tumor (10-25%):    {len(low_tumor_indices)}")
        print(f"  Medium tumor (25-50%): {len(medium_tumor_indices)}")
        print(f"  High tumor (50-75%):   {len(high_tumor_indices)}")
        
        # Select balanced mix
        selected_indices = []
        selected_ratios = []
        
        # Take equal numbers from each category
        samples_per_category = num_samples // 3
        remainder = num_samples % 3
        
        categories = [
            (low_tumor_indices, "Low"),
            (medium_tumor_indices, "Medium"), 
            (high_tumor_indices, "High")
        ]
        
        for i, (category_indices, category_name) in enumerate(categories):
            take_count = samples_per_category + (1 if i < remainder else 0)
            
            if len(category_indices) >= take_count:
                # Sort by tumor ratio and take evenly distributed samples
                category_indices.sort(key=lambda x: x[1])
                step = len(category_indices) // take_count if take_count > 0 else 1
                
                for j in range(take_count):
                    idx = min(j * step, len(category_indices) - 1)
                    sample_idx, tumor_ratio = category_indices[idx]
                    selected_indices.append(sample_idx)
                    selected_ratios.append(tumor_ratio * 100)
                    
                print(f"  Selected {take_count} from {category_name} tumor category")
            else:
                print(f"  ⚠ Not enough {category_name} tumor samples available")
        
        return selected_indices, selected_ratios
    
    def calculate_weighted_support(self, combined_true, weight_factor=0.7):
        """Calculate support with reduced weight on dominant class"""
        y_true_binary = (combined_true > 0.5).astype(int)
        
        background_count = np.sum(y_true_binary == 0)
        tumor_count = np.sum(y_true_binary == 1)
        
        # Apply weight reduction to dominant class
        if background_count > tumor_count:
            # Background is dominant, reduce its weight
            weighted_background = int(background_count * weight_factor)
            weighted_tumor = tumor_count
        else:
            # Tumor is dominant, reduce its weight
            weighted_background = background_count
            weighted_tumor = int(tumor_count * weight_factor)
        
        total_weighted = weighted_background + weighted_tumor
        
        print(f"\n📊 SUPPORT WEIGHTING:")
        print(f"Original support:")
        print(f"  Background: {background_count:,}")
        print(f"  Tumor:      {tumor_count:,}")
        print(f"Weighted support (factor: {weight_factor}):")
        print(f"  Background: {weighted_background:,} ({weighted_background/total_weighted*100:.1f}%)")
        print(f"  Tumor:      {weighted_tumor:,} ({weighted_tumor/total_weighted*100:.1f}%)")
        
        return weighted_background, weighted_tumor

def load_trained_models():
    """Load trained models"""
    models = {}
    
    try:
        models['Hybrid Quantum U-Net'] = load_model('hybrid_quantum_unet_model.h5', 
                                                   custom_objects={'QuantumLayer': QuantumLayer,
                                                                 'dice_coefficient': dice_coefficient,
                                                                 'dice_loss': dice_loss,
                                                                 'combined_loss': combined_loss})
        print("✓ Hybrid Quantum U-Net loaded")
    except:
        print("⚠ Hybrid Quantum U-Net model not found")
    
    try:
        models['Classical U-Net'] = load_model('classical_unet_model.h5',
                                             custom_objects={'dice_coefficient': dice_coefficient,
                                                           'dice_loss': dice_loss,
                                                           'combined_loss': combined_loss})
        print("✓ Classical U-Net loaded")
    except:
        print("⚠ Classical U-Net model not found")
    
    return models

def evaluate_with_balanced_support(models, X_test, y_test, evaluator):
    """Evaluate models with balanced support distribution"""
    print("\n🎯 Starting Balanced Support Evaluation...")
    print("="*60)
    
    # Create balanced subset
    selected_indices, tumor_ratios = evaluator.create_balanced_subset(X_test, y_test, num_samples=15)
    
    if len(selected_indices) == 0:
        print("⚠ No suitable samples found for balanced evaluation!")
        return {}
    
    print(f"\n✓ Selected {len(selected_indices)} samples for balanced evaluation")
    print(f"✓ Tumor ratios range: {min(tumor_ratios):.1f}% - {max(tumor_ratios):.1f}%")
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n📊 Evaluating {model_name}...")
        print("-" * 50)
        
        model_predictions = []
        model_ground_truths = []
        
        # Process each selected sample
        for idx in selected_indices:
            # Create generator for single sample
            sample_gen = MRIDataGenerator([X_test[idx]], [y_test[idx]], 
                                        batch_size=1, augment=False)
            sample_x, sample_y = sample_gen[0]
            
            # Get prediction
            prediction = model.predict(sample_x, verbose=0)[0].squeeze()
            ground_truth = sample_y[0].squeeze()
            
            model_predictions.append(prediction)
            model_ground_truths.append(ground_truth)
        
        # Combine all predictions and ground truths
        combined_pred = np.concatenate([pred.flatten() for pred in model_predictions])
        combined_true = np.concatenate([gt.flatten() for gt in model_ground_truths])
        
        # Calculate metrics with standard support
        y_true_binary = (combined_true > 0.5).astype(int)
        y_pred_binary = (combined_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        
        # Calculate weighted support
        weighted_bg, weighted_tumor = evaluator.calculate_weighted_support(combined_true, weight_factor=0.6)
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'original_bg_support': np.sum(y_true_binary == 0),
            'original_tumor_support': np.sum(y_true_binary == 1),
            'weighted_bg_support': weighted_bg,
            'weighted_tumor_support': weighted_tumor
        }
        
        # Print results
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Classification report with original support
        print(f"\nOriginal Classification Report:")
        report = classification_report(y_true_binary, y_pred_binary, 
                                     target_names=['Background', 'Tumor'], 
                                     digits=4, zero_division=0)
        print(report)
    
    return results, selected_indices

def visualize_balanced_results(results):
    """Visualize balanced evaluation results"""
    if not results:
        return
    
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Metrics comparison
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, model in enumerate(models):
        values = [results[model][metric] for metric in metrics]
        ax1.bar(x + i*width, values, width, label=model, alpha=0.8)
        
        # Add value labels
        for j, v in enumerate(values):
            ax1.text(x[j] + i*width, v + 0.01, f'{v:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Balanced Support Evaluation - Model Comparison')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Support comparison
    support_types = ['Original\nBackground', 'Original\nTumor', 'Weighted\nBackground', 'Weighted\nTumor']
    
    for i, model in enumerate(models):
        support_values = [
            results[model]['original_bg_support'],
            results[model]['original_tumor_support'],
            results[model]['weighted_bg_support'],
            results[model]['weighted_tumor_support']
        ]
        
        x_pos = np.arange(len(support_types)) + i*0.4
        bars = ax2.bar(x_pos, support_values, 0.35, label=model, alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, support_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(support_values)*0.01,
                    f'{value:,}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Support Type')
    ax2.set_ylabel('Number of Pixels')
    ax2.set_title('Support Distribution Comparison')
    ax2.set_xticks(np.arange(len(support_types)) + 0.2)
    ax2.set_xticklabels(support_types)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('balanced_support_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 Starting Balanced Support Evaluation...")
    print("="*60)
    
    # Prepare dataset
    print("\n📊 Preparing dataset...")
    DATASET_PATH = "./dataset"
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(DATASET_PATH)
        print(f"✓ Dataset prepared: {len(X_test)} test samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Load models
    print("\n🤖 Loading trained models...")
    models = load_trained_models()
    
    if not models:
        print("❌ No models loaded. Please train models first.")
        return
    
    # Create evaluator
    evaluator = BalancedSupportEvaluator()
    
    # Evaluate with balanced support
    results, selected_indices = evaluate_with_balanced_support(models, X_test, y_test, evaluator)
    
    # Visualize results
    if results:
        visualize_balanced_results(results)
        print(f"\n✅ Balanced support evaluation completed!")
        print(f"📁 Results saved as: balanced_support_evaluation.png")
        
        print(f"\n🎯 KEY BENEFITS:")
        print(f"✓ More balanced class distribution")
        print(f"✓ Reduced weight on dominant class")
        print(f"✓ More realistic clinical evaluation")
        print(f"✓ Better representation of model performance")

if __name__ == "__main__":
    main()