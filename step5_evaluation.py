"""
Step 5: Model Evaluation
Comprehensive Evaluation of Hybrid Quantum U-Net vs Classical U-Net
for Lower-grade Glioma Segmentation in MRI
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from pathlib import Path

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

class SegmentationEvaluator:
    
    def __init__(self):
        self.metrics_history = []
        
    def dice_coefficient_np(self, y_true, y_pred, smooth=1e-6):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    def iou_score(self, y_true, y_pred, smooth=1e-6):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)
    
    def sensitivity_specificity(self, y_true, y_pred):
        y_true_f = y_true.flatten().astype(int)
        y_pred_f = (y_pred.flatten() > 0.5).astype(int)
        
        tp = np.sum((y_true_f == 1) & (y_pred_f == 1))
        tn = np.sum((y_true_f == 0) & (y_pred_f == 0))
        fp = np.sum((y_true_f == 0) & (y_pred_f == 1))
        fn = np.sum((y_true_f == 1) & (y_pred_f == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return sensitivity, specificity
    
    def hausdorff_distance(self, y_true, y_pred):
        try:
            y_true_bin = (y_true > 0.5).astype(np.uint8)
            y_pred_bin = (y_pred > 0.5).astype(np.uint8)
            
            contours_true, _ = cv2.findContours(y_true_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_pred, _ = cv2.findContours(y_pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours_true) == 0 or len(contours_pred) == 0:
                return float('inf')
            
            contour_true = max(contours_true, key=cv2.contourArea)
            contour_pred = max(contours_pred, key=cv2.contourArea)
            
            points_true = contour_true.reshape(-1, 2)
            points_pred = contour_pred.reshape(-1, 2)
            
            hd1 = directed_hausdorff(points_true, points_pred)[0]
            hd2 = directed_hausdorff(points_pred, points_true)[0]
            
            return max(hd1, hd2)
        except:
            return float('inf')
    
    def evaluate_sample(self, y_true, y_pred):
        dice = self.dice_coefficient_np(y_true, y_pred)
        iou = self.iou_score(y_true, y_pred)
        sensitivity, specificity = self.sensitivity_specificity(y_true, y_pred)
        hausdorff = self.hausdorff_distance(y_true, y_pred)
        
        return {
            'dice': dice,
            'iou': iou,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'hausdorff': hausdorff
        }
    
    def evaluate_model(self, model, test_generator, model_name):
        print(f"\n🔍 Evaluating {model_name}...")
        
        all_metrics = []
        predictions = []
        ground_truths = []
        all_y_true_flat = []
        all_y_pred_flat = []
        
        for i in range(len(test_generator)):
            batch_x, batch_y = test_generator[i]
            batch_pred = model.predict(batch_x, verbose=0)
            
            for j in range(len(batch_x)):
                y_true = batch_y[j].squeeze()
                y_pred = batch_pred[j].squeeze()
                
                y_true_flat = (y_true.flatten() > 0.5).astype(int)
                y_pred_flat = (y_pred.flatten() > 0.5).astype(int)
                all_y_true_flat.extend(y_true_flat)
                all_y_pred_flat.extend(y_pred_flat)
                
                metrics = self.evaluate_sample(y_true, y_pred)
                metrics['model'] = model_name
                metrics['sample_id'] = i * test_generator.batch_size + j
                
                all_metrics.append(metrics)
                predictions.append(y_pred)
                ground_truths.append(y_true)
        
        print(f"\n📊 Classification Report for {model_name}:")
        print("="*50)
        report = classification_report(
            all_y_true_flat, 
            all_y_pred_flat,
            target_names=['Background', 'Tumor'],
            digits=4
        )
        print(report)
        
        return pd.DataFrame(all_metrics), predictions, ground_truths, report

class VisualizationManager:
    
    def __init__(self, output_dir="evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_metrics_comparison(self, df_results):
        metrics = ['dice', 'iou', 'sensitivity', 'specificity']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            sns.boxplot(data=df_results, x='model', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.capitalize()} Score Comparison', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(f'{metric.capitalize()} Score')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_segmentation_examples(self, images, ground_truths, pred_hybrid, pred_classical, n_examples=6):
        fig, axes = plt.subplots(n_examples, 4, figsize=(16, 4*n_examples))
        
        for i in range(n_examples):
            axes[i, 0].imshow(images[i].squeeze(), cmap='gray')
            axes[i, 0].set_title('Original MRI')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(ground_truths[i], cmap='Reds', alpha=0.7)
            axes[i, 1].imshow(images[i].squeeze(), cmap='gray', alpha=0.3)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_hybrid[i], cmap='Blues', alpha=0.7)
            axes[i, 2].imshow(images[i].squeeze(), cmap='gray', alpha=0.3)
            axes[i, 2].set_title('Hybrid Quantum U-Net')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_classical[i], cmap='Greens', alpha=0.7)
            axes[i, 3].imshow(images[i].squeeze(), cmap='gray', alpha=0.3)
            axes[i, 3].set_title('Classical U-Net')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'segmentation_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_statistics(self, df_results):
        stats = df_results.groupby('model').agg({
            'dice': ['mean', 'std', 'min', 'max'],
            'iou': ['mean', 'std', 'min', 'max'],
            'sensitivity': ['mean', 'std', 'min', 'max'],
            'specificity': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        stats.columns = [f'{col[0]}_{col[1]}' for col in stats.columns]
        
        sns.heatmap(stats, annot=True, cmap='RdYlBu_r', center=0.5, ax=ax)
        ax.set_title('Performance Statistics Heatmap', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats

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
            print("Creating new model for evaluation...")
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
            print("Creating new model for evaluation...")
            models['Classical U-Net'] = create_classical_unet()
    else:
        print("⚠ Classical model not found, creating new model...")
        models['Classical U-Net'] = create_classical_unet()
    
    return models

def generate_evaluation_report(df_results, stats, output_dir="evaluation_results"):
    output_path = Path(output_dir) / 'evaluation_report.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HYBRID QUANTUM U-NET vs CLASSICAL U-NET EVALUATION REPORT\n")
        f.write("Lower-grade Glioma Segmentation in MRI\n")
        f.write("="*80 + "\n\n")
        
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-"*40 + "\n")
        
        for model in df_results['model'].unique():
            model_data = df_results[df_results['model'] == model]
            f.write(f"\n{model}:\n")
            f.write(f"  Dice Coefficient: {model_data['dice'].mean():.4f} ± {model_data['dice'].std():.4f}\n")
            f.write(f"  IoU Score:        {model_data['iou'].mean():.4f} ± {model_data['iou'].std():.4f}\n")
            f.write(f"  Sensitivity:      {model_data['sensitivity'].mean():.4f} ± {model_data['sensitivity'].std():.4f}\n")
            f.write(f"  Specificity:      {model_data['specificity'].mean():.4f} ± {model_data['specificity'].std():.4f}\n")
            
            valid_hd = model_data[model_data['hausdorff'] != float('inf')]['hausdorff']
            if len(valid_hd) > 0:
                f.write(f"  Hausdorff Dist:   {valid_hd.mean():.4f} ± {valid_hd.std():.4f}\n")
            else:
                f.write(f"  Hausdorff Dist:   N/A (no valid contours)\n")
        
        f.write("\n\nSTATISTICAL ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        hybrid_data = df_results[df_results['model'] == 'Hybrid Quantum U-Net']
        classical_data = df_results[df_results['model'] == 'Classical U-Net']
        
        for metric in ['dice', 'iou', 'sensitivity', 'specificity']:
            hybrid_mean = hybrid_data[metric].mean()
            classical_mean = classical_data[metric].mean()
            improvement = ((hybrid_mean - classical_mean) / classical_mean) * 100
            
            f.write(f"\n{metric.capitalize()}:\n")
            f.write(f"  Hybrid:    {hybrid_mean:.4f}\n")
            f.write(f"  Classical: {classical_mean:.4f}\n")
            f.write(f"  Improvement: {improvement:+.2f}%\n")
        
        f.write("\n\nCONCLUSIONS\n")
        f.write("-"*40 + "\n")
        f.write("The evaluation demonstrates the comparative performance of hybrid quantum\n")
        f.write("and classical U-Net architectures for glioma segmentation. Results show\n")
        f.write("the potential benefits and limitations of quantum-enhanced deep learning\n")
        f.write("approaches in medical image segmentation tasks.\n")
    
    print(f"✓ Evaluation report saved to: {output_path}")

def main():
    print("🚀 Starting Comprehensive Model Evaluation...")
    print("="*60)
    
    print("\n📊 Preparing test dataset...")
    DATASET_PATH = "./dataset"
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(DATASET_PATH)
    print(f"✓ Test set prepared: {len(X_test)} samples")
    
    test_gen = MRIDataGenerator(X_test, y_test, batch_size=8, augment=False)
    
    print("\n🤖 Loading trained models...")
    models = load_trained_models()
    
    evaluator = SegmentationEvaluator()
    visualizer = VisualizationManager()
    
    all_results = []
    all_predictions = {}
    all_ground_truths = None
    all_images = None
    
    for model_name, model in models.items():
        df_metrics, predictions, ground_truths, report = evaluator.evaluate_model(
            model, test_gen, model_name
        )
        all_results.append(df_metrics)
        all_predictions[model_name] = predictions
        
        if all_ground_truths is None:
            all_ground_truths = ground_truths
            all_images = []
            for i in range(min(6, len(test_gen))):
                batch_x, _ = test_gen[i]
                all_images.extend(batch_x[:min(6-len(all_images), len(batch_x))])
    
    df_combined = pd.concat(all_results, ignore_index=True)
    
    print("\n📈 Generating visualizations...")
    visualizer.plot_metrics_comparison(df_combined)
    
    if len(all_images) >= 6 and 'Hybrid Quantum U-Net' in all_predictions and 'Classical U-Net' in all_predictions:
        visualizer.plot_segmentation_examples(
            all_images[:6], 
            all_ground_truths[:6],
            all_predictions['Hybrid Quantum U-Net'][:6],
            all_predictions['Classical U-Net'][:6]
        )
    
    stats = visualizer.plot_performance_statistics(df_combined)
    
    print("\n📝 Generating evaluation report...")
    generate_evaluation_report(df_combined, stats)
    
    print("\n" + "="*60)
    print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nSUMMARY RESULTS:")
    for model in df_combined['model'].unique():
        model_data = df_combined[df_combined['model'] == model]
        print(f"\n{model}:")
        print(f"  Average Dice Score: {model_data['dice'].mean():.4f}")
        print(f"  Average IoU Score:  {model_data['iou'].mean():.4f}")
    
    print(f"\n📁 All results saved to: evaluation_results/")
    print("   - metrics_comparison.png")
    print("   - segmentation_examples.png") 
    print("   - performance_heatmap.png")
    print("   - evaluation_report.txt")

if __name__ == "__main__":
    main()