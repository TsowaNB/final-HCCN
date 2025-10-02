"""
Sample 590 Support Calculator
Calculate exact support values for your current test (sample 590)
"""

import os
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from step2_preprocessing import MRIPreprocessor, MRIDataGenerator, prepare_dataset

def calculate_sample_590_support():
    """Calculate exact support values for sample 590"""
    print("🔍 CALCULATING SUPPORT FOR SAMPLE 590")
    print("="*60)
    
    # Prepare dataset
    print("\n📊 Loading dataset...")
    DATASET_PATH = "./dataset"
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(DATASET_PATH)
        print(f"✓ Dataset loaded: {len(X_test)} test samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Check if sample 590 exists
    sample_index = 590
    if sample_index >= len(X_test):
        print(f"⚠ Sample {sample_index} not found. Dataset has {len(X_test)} samples.")
        sample_index = min(sample_index, len(X_test) - 1)
        print(f"Using sample {sample_index} instead.")
    
    print(f"\n🎯 ANALYZING SAMPLE {sample_index}")
    print("-" * 40)
    
    # Get the ground truth mask for sample 590
    ground_truth_mask = y_test[sample_index]
    
    print(f"Image dimensions: {ground_truth_mask.shape}")
    
    # Convert to binary (0 = background, 1 = tumor)
    binary_mask = (ground_truth_mask > 0.5).astype(int)
    
    # Calculate support values
    total_pixels = binary_mask.size
    background_pixels = np.sum(binary_mask == 0)
    tumor_pixels = np.sum(binary_mask == 1)
    
    background_percentage = (background_pixels / total_pixels) * 100
    tumor_percentage = (tumor_pixels / total_pixels) * 100
    
    print(f"\n✅ SUPPORT CALCULATION FOR SAMPLE {sample_index}:")
    print(f"Total pixels:        {total_pixels:,}")
    print(f"Background pixels:   {background_pixels:,} ({background_percentage:.2f}%)")
    print(f"Tumor pixels:        {tumor_pixels:,} ({tumor_percentage:.2f}%)")
    
    # Show the actual support values that would appear in classification report
    print(f"\n📋 CLASSIFICATION REPORT SUPPORT VALUES:")
    print(f"Background support = {background_pixels}")
    print(f"Tumor support      = {tumor_pixels}")
    
    # Create a dummy prediction to show classification report format
    dummy_prediction = np.random.choice([0, 1], size=binary_mask.shape, 
                                      p=[0.6, 0.4])  # Random prediction
    
    print(f"\n📊 SAMPLE CLASSIFICATION REPORT:")
    print("(Using dummy predictions - support values are what matter)")
    report = classification_report(binary_mask.flatten(), dummy_prediction.flatten(),
                                 target_names=['Background', 'Tumor'], 
                                 digits=4)
    print(report)
    
    # Determine if this is tumor-rich
    min_tumor_threshold = 500
    is_tumor_rich = tumor_pixels >= min_tumor_threshold
    
    print(f"\n🎯 TUMOR-RICH ANALYSIS:")
    print(f"Minimum tumor pixels for 'tumor-rich': {min_tumor_threshold}")
    print(f"Sample {sample_index} tumor pixels: {tumor_pixels}")
    print(f"Is tumor-rich? {'✅ YES' if is_tumor_rich else '❌ NO'}")
    
    if is_tumor_rich:
        print(f"✓ This sample WILL be included in enhanced evaluation")
    else:
        print(f"⚠ This sample will NOT be included in enhanced evaluation")
        print(f"  (Need ≥{min_tumor_threshold} tumor pixels)")
    
    return {
        'sample_index': sample_index,
        'total_pixels': total_pixels,
        'background_support': background_pixels,
        'tumor_support': tumor_pixels,
        'background_percentage': background_percentage,
        'tumor_percentage': tumor_percentage,
        'is_tumor_rich': is_tumor_rich
    }

def compare_with_enhanced_evaluation():
    """Show how sample 590 compares to enhanced evaluation approach"""
    print(f"\n" + "="*60)
    print("🔄 COMPARISON: SINGLE SAMPLE vs ENHANCED EVALUATION")
    print("="*60)
    
    sample_info = calculate_sample_590_support()
    
    print(f"\n📊 SINGLE SAMPLE {sample_info['sample_index']} SUPPORT:")
    print(f"Background: {sample_info['background_support']:,} ({sample_info['background_percentage']:.1f}%)")
    print(f"Tumor:      {sample_info['tumor_support']:,} ({sample_info['tumor_percentage']:.1f}%)")
    
    print(f"\n📊 ENHANCED EVALUATION (20 tumor-rich samples):")
    print(f"Estimated total pixels: 20 × 256 × 256 = 1,310,720")
    print(f"Estimated tumor support: ~800,000-900,000 (60-70%)")
    print(f"Estimated background support: ~400,000-500,000 (30-40%)")
    
    print(f"\n🎯 KEY DIFFERENCES:")
    if sample_info['is_tumor_rich']:
        print(f"✅ Sample {sample_info['sample_index']} IS tumor-rich and would be included")
        print(f"✅ Enhanced evaluation combines 20 such samples")
        print(f"✅ Result: Much higher tumor support for realistic evaluation")
    else:
        print(f"❌ Sample {sample_info['sample_index']} is NOT tumor-rich")
        print(f"❌ Enhanced evaluation would skip this sample")
        print(f"✅ Enhanced evaluation only uses samples with significant tumors")

if __name__ == "__main__":
    try:
        sample_info = calculate_sample_590_support()
        compare_with_enhanced_evaluation()
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Run: python step7_enhanced_evaluation.py")
        print(f"2. This will show tumor-dominated support values")
        print(f"3. Much more clinically relevant than single sample testing!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Make sure your dataset is properly set up in ./dataset/")