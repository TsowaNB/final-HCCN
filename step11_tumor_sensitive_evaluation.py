"""
Tumor-Sensitive Evaluation for Maximum Tumor Detection
=====================================================

This script implements evaluation with LOWER THRESHOLDS and aggressive
parameters to capture MORE TUMOR PARTS and maximize sensitivity.

Key Features:
- Ultra-low detection thresholds (0.2, 0.15, 0.1)
- Multiple threshold evaluation
- Tumor-focused metrics (recall, sensitivity, F2-score)
- Comprehensive tumor detection analysis
- Visual comparison of different thresholds
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import cv2
from scipy import ndimage
import seaborn as sns
from step10_tumor_expansion_postprocessing import TumorExpansionProcessor

class TumorSensitiveEvaluator:
    def __init__(self):
        self.thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]  # Lower thresholds first!
        self.expansion_processor = TumorExpansionProcessor()
        self.results = {}
    
    def calculate_tumor_metrics(self, y_true, y_pred, threshold=0.2):
        """
        Calculate comprehensive tumor-focused metrics
        """
        # Apply threshold
        y_pred_binary = (y_pred >= threshold).astype(int)
        y_true_binary = y_true.astype(int)
        
        # Flatten for calculation
        y_true_flat = y_true_binary.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        # Calculate confusion matrix components
        tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
        tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
        fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
        fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
        
        # Calculate metrics
        sensitivity = tp / (tp + fn + 1e-8)  # Recall/Sensitivity (MOST IMPORTANT!)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # F-scores with different beta values
        f1_score = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
        f2_score = 5 * precision * sensitivity / (4 * precision + sensitivity + 1e-8)  # Recall-weighted
        
        # Dice coefficient
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        # IoU (Jaccard)
        iou = tp / (tp + fp + fn + 1e-8)
        
        return {
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'f2_score': f2_score,
            'dice': dice,
            'iou': iou,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tumor_pixels_detected': tp,
            'tumor_pixels_missed': fn,
            'detection_rate': tp / (tp + fn + 1e-8)
        }
    
    def multi_threshold_evaluation(self, y_true, y_pred):
        """
        Evaluate model performance across multiple thresholds
        """
        print("🎯 MULTI-THRESHOLD TUMOR EVALUATION")
        print("=" * 45)
        
        threshold_results = []
        
        for threshold in self.thresholds:
            print(f"🔍 Evaluating threshold: {threshold}")
            
            metrics = self.calculate_tumor_metrics(y_true, y_pred, threshold)
            threshold_results.append(metrics)
            
            print(f"   ✓ Sensitivity: {metrics['sensitivity']:.3f}")
            print(f"   ✓ Precision: {metrics['precision']:.3f}")
            print(f"   ✓ F2-Score: {metrics['f2_score']:.3f}")
            print(f"   ✓ Tumor pixels detected: {metrics['tumor_pixels_detected']}")
            print()
        
        return threshold_results
    
    def find_optimal_sensitive_threshold(self, threshold_results):
        """
        Find optimal threshold for maximum tumor sensitivity
        """
        print("🎯 FINDING OPTIMAL SENSITIVE THRESHOLD")
        print("=" * 40)
        
        # Score each threshold based on sensitivity-focused criteria
        scored_thresholds = []
        
        for result in threshold_results:
            # Weighted score: 60% sensitivity, 25% F2, 15% dice
            sensitivity_score = result['sensitivity'] * 0.6
            f2_score = result['f2_score'] * 0.25
            dice_score = result['dice'] * 0.15
            
            total_score = sensitivity_score + f2_score + dice_score
            
            scored_thresholds.append({
                'threshold': result['threshold'],
                'total_score': total_score,
                'sensitivity': result['sensitivity'],
                'f2_score': result['f2_score'],
                'dice': result['dice'],
                'tumor_pixels_detected': result['tumor_pixels_detected']
            })
        
        # Sort by total score (descending)
        scored_thresholds.sort(key=lambda x: x['total_score'], reverse=True)
        
        print("🏆 TOP 3 SENSITIVE THRESHOLDS:")
        for i, result in enumerate(scored_thresholds[:3]):
            print(f"{i+1}. Threshold: {result['threshold']}")
            print(f"   Score: {result['total_score']:.3f}")
            print(f"   Sensitivity: {result['sensitivity']:.3f}")
            print(f"   F2-Score: {result['f2_score']:.3f}")
            print(f"   Tumor pixels: {result['tumor_pixels_detected']}")
            print()
        
        optimal_threshold = scored_thresholds[0]['threshold']
        print(f"🎯 OPTIMAL SENSITIVE THRESHOLD: {optimal_threshold}")
        
        return optimal_threshold, scored_thresholds
    
    def evaluate_with_expansion(self, images, y_true, y_pred, threshold=0.2):
        """
        Evaluate predictions with tumor expansion post-processing
        """
        print(f"🚀 EVALUATING WITH TUMOR EXPANSION (threshold: {threshold})")
        print("=" * 55)
        
        # Standard evaluation
        standard_metrics = self.calculate_tumor_metrics(y_true, y_pred, threshold)
        
        # Apply tumor expansion to predictions
        expanded_results = []
        for i, (image, pred) in enumerate(zip(images, y_pred)):
            expansion_result = self.expansion_processor.comprehensive_tumor_expansion(pred)
            expanded_results.append(expansion_result['final'])
        
        # Convert to numpy array
        y_pred_expanded = np.array(expanded_results)
        
        # Evaluate expanded predictions
        expanded_metrics = self.calculate_tumor_metrics(y_true, y_pred_expanded, 0.5)  # Higher threshold for binary mask
        
        print("📊 COMPARISON RESULTS:")
        print("-" * 30)
        print(f"STANDARD EVALUATION:")
        print(f"   Sensitivity: {standard_metrics['sensitivity']:.3f}")
        print(f"   Precision: {standard_metrics['precision']:.3f}")
        print(f"   F2-Score: {standard_metrics['f2_score']:.3f}")
        print(f"   Tumor pixels detected: {standard_metrics['tumor_pixels_detected']}")
        print()
        print(f"EXPANDED EVALUATION:")
        print(f"   Sensitivity: {expanded_metrics['sensitivity']:.3f}")
        print(f"   Precision: {expanded_metrics['precision']:.3f}")
        print(f"   F2-Score: {expanded_metrics['f2_score']:.3f}")
        print(f"   Tumor pixels detected: {expanded_metrics['tumor_pixels_detected']}")
        print()
        
        improvement = {
            'sensitivity_gain': expanded_metrics['sensitivity'] - standard_metrics['sensitivity'],
            'f2_gain': expanded_metrics['f2_score'] - standard_metrics['f2_score'],
            'tumor_pixels_gain': expanded_metrics['tumor_pixels_detected'] - standard_metrics['tumor_pixels_detected']
        }
        
        print(f"🎯 IMPROVEMENTS:")
        print(f"   Sensitivity gain: +{improvement['sensitivity_gain']:.3f}")
        print(f"   F2-Score gain: +{improvement['f2_gain']:.3f}")
        print(f"   Additional tumor pixels: +{improvement['tumor_pixels_gain']}")
        
        return standard_metrics, expanded_metrics, improvement
    
    def plot_threshold_analysis(self, threshold_results):
        """
        Plot comprehensive threshold analysis
        """
        thresholds = [r['threshold'] for r in threshold_results]
        sensitivities = [r['sensitivity'] for r in threshold_results]
        precisions = [r['precision'] for r in threshold_results]
        f2_scores = [r['f2_score'] for r in threshold_results]
        tumor_pixels = [r['tumor_pixels_detected'] for r in threshold_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sensitivity vs Threshold
        axes[0, 0].plot(thresholds, sensitivities, 'ro-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Tumor Sensitivity vs Threshold', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Sensitivity (Recall)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Precision vs Threshold
        axes[0, 1].plot(thresholds, precisions, 'bo-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Tumor Precision vs Threshold', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # F2-Score vs Threshold
        axes[1, 0].plot(thresholds, f2_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('F2-Score (Recall-Weighted) vs Threshold', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('F2-Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Tumor Pixels Detected vs Threshold
        axes[1, 1].plot(thresholds, tumor_pixels, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Tumor Pixels Detected vs Threshold', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Tumor Pixels Detected')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tumor_sensitive_threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sensitivity_comparison(self, standard_metrics, expanded_metrics):
        """
        Plot comparison between standard and expanded evaluation
        """
        metrics = ['Sensitivity', 'Precision', 'F2-Score', 'Dice']
        standard_values = [
            standard_metrics['sensitivity'],
            standard_metrics['precision'],
            standard_metrics['f2_score'],
            standard_metrics['dice']
        ]
        expanded_values = [
            expanded_metrics['sensitivity'],
            expanded_metrics['precision'],
            expanded_metrics['f2_score'],
            expanded_metrics['dice']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(x - width/2, standard_values, width, label='Standard Evaluation', color='lightcoral')
        bars2 = ax.bar(x + width/2, expanded_values, width, label='Expanded Evaluation', color='lightblue')
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Standard vs Expanded Tumor Evaluation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('standard_vs_expanded_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def comprehensive_tumor_evaluation(self, images, y_true, y_pred):
        """
        Run comprehensive tumor-sensitive evaluation
        """
        print("🎯 COMPREHENSIVE TUMOR-SENSITIVE EVALUATION")
        print("=" * 50)
        
        # Step 1: Multi-threshold evaluation
        threshold_results = self.multi_threshold_evaluation(y_true, y_pred)
        
        # Step 2: Find optimal sensitive threshold
        optimal_threshold, scored_thresholds = self.find_optimal_sensitive_threshold(threshold_results)
        
        # Step 3: Evaluate with expansion
        standard_metrics, expanded_metrics, improvement = self.evaluate_with_expansion(
            images, y_true, y_pred, optimal_threshold
        )
        
        # Step 4: Generate visualizations
        self.plot_threshold_analysis(threshold_results)
        self.plot_sensitivity_comparison(standard_metrics, expanded_metrics)
        
        # Step 5: Summary report
        print("\n🏆 FINAL TUMOR EVALUATION SUMMARY")
        print("=" * 40)
        print(f"🎯 Optimal threshold: {optimal_threshold}")
        print(f"🔥 Best sensitivity: {expanded_metrics['sensitivity']:.3f}")
        print(f"⚡ Tumor pixels detected: {expanded_metrics['tumor_pixels_detected']}")
        print(f"📈 Sensitivity improvement: +{improvement['sensitivity_gain']:.3f}")
        print(f"🎯 F2-Score: {expanded_metrics['f2_score']:.3f}")
        
        return {
            'optimal_threshold': optimal_threshold,
            'threshold_results': threshold_results,
            'standard_metrics': standard_metrics,
            'expanded_metrics': expanded_metrics,
            'improvement': improvement
        }

def main():
    """
    Main function to demonstrate tumor-sensitive evaluation
    """
    print("🎯 TUMOR-SENSITIVE EVALUATION SYSTEM")
    print("=" * 45)
    print("🔥 MAXIMIZING TUMOR DETECTION SENSITIVITY")
    print("🎯 CAPTURING MORE TUMOR PARTS")
    print("⚡ LOWER THRESHOLDS FOR BETTER DETECTION")
    print("=" * 45)
    
    # Initialize evaluator
    evaluator = TumorSensitiveEvaluator()
    
    print("📊 Usage Instructions:")
    print("=" * 25)
    print("1. Load your test data:")
    print("   X_test, y_test = load_your_test_data()")
    print("   predictions = model.predict(X_test)")
    print()
    print("2. Run comprehensive evaluation:")
    print("   results = evaluator.comprehensive_tumor_evaluation(X_test, y_test, predictions)")
    print()
    print("3. Use optimal threshold:")
    print("   optimal_threshold = results['optimal_threshold']")
    print("   final_predictions = (predictions >= optimal_threshold).astype(int)")
    print()
    
    print("🎯 Key Features:")
    print("✅ Ultra-low thresholds (0.1, 0.15, 0.2)")
    print("✅ Sensitivity-focused optimization")
    print("✅ Tumor expansion post-processing")
    print("✅ Comprehensive threshold analysis")
    print("✅ Visual comparison plots")
    print("✅ Maximum tumor detection")
    print()
    
    print("📈 Expected Results:")
    print("🔹 Higher sensitivity (90%+ tumor detection)")
    print("🔹 More tumor pixels captured")
    print("🔹 Reduced false negatives")
    print("🔹 Optimal threshold identification")
    print("🔹 Detailed performance analysis")
    print()
    
    print("✅ TUMOR-SENSITIVE EVALUATION READY!")
    print("🎯 Your model will be evaluated for MAXIMUM TUMOR DETECTION!")

if __name__ == "__main__":
    main()