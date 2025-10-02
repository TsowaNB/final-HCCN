"""
Tumor Expansion Post-Processing for Maximum Tumor Detection
==========================================================

This script implements advanced post-processing techniques to EXPAND
tumor predictions and capture MORE TUMOR PARTS that might be missed.

Key Features:
- Morphological expansion operations
- Region growing algorithms
- Tumor boundary expansion
- Connected component analysis
- Aggressive tumor detection thresholds
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, measure, segmentation
from skimage.feature import peak_local_maxima
import matplotlib.pyplot as plt

class TumorExpansionProcessor:
    def __init__(self):
        self.expansion_params = {
            'detection_threshold': 0.25,  # Lower threshold for initial detection
            'expansion_threshold': 0.15,  # Even lower for expansion
            'min_tumor_size': 50,         # Minimum tumor size in pixels
            'expansion_iterations': 3,    # Number of expansion iterations
            'closing_kernel_size': 5,     # Morphological closing kernel
            'opening_kernel_size': 3      # Morphological opening kernel
        }
    
    def aggressive_threshold_detection(self, prediction, threshold=0.25):
        """
        Apply aggressive (lower) thresholding for initial tumor detection
        """
        # Primary detection with lower threshold
        primary_mask = (prediction >= threshold).astype(np.uint8)
        
        # Secondary detection for weak signals
        secondary_mask = (prediction >= threshold * 0.6).astype(np.uint8)
        
        return primary_mask, secondary_mask
    
    def morphological_tumor_expansion(self, mask, iterations=3):
        """
        Expand tumor regions using morphological operations
        """
        # Define expansion kernels
        expansion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Step 1: Close small gaps within tumors
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        
        # Step 2: Iterative dilation to expand tumor boundaries
        expanded_mask = closed_mask.copy()
        for i in range(iterations):
            expanded_mask = cv2.dilate(expanded_mask, expansion_kernel, iterations=1)
        
        # Step 3: Fill holes within expanded regions
        filled_mask = ndimage.binary_fill_holes(expanded_mask).astype(np.uint8)
        
        return filled_mask
    
    def region_growing_expansion(self, prediction, seed_mask, growth_threshold=0.15):
        """
        Expand tumor regions using region growing algorithm
        """
        # Initialize grown mask with seed regions
        grown_mask = seed_mask.copy()
        
        # Get image dimensions
        h, w = prediction.shape
        
        # Define 8-connectivity neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # Iterative region growing
        changed = True
        iteration = 0
        max_iterations = 10
        
        while changed and iteration < max_iterations:
            changed = False
            new_grown_mask = grown_mask.copy()
            
            # Find boundary pixels of current grown regions
            boundary = cv2.dilate(grown_mask, np.ones((3, 3), np.uint8)) - grown_mask
            boundary_coords = np.where(boundary > 0)
            
            for y, x in zip(boundary_coords[0], boundary_coords[1]):
                # Check if this boundary pixel should be included
                if prediction[y, x] >= growth_threshold:
                    # Check if it's connected to existing tumor region
                    connected = False
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if grown_mask[ny, nx] > 0:
                                connected = True
                                break
                    
                    if connected:
                        new_grown_mask[y, x] = 1
                        changed = True
            
            grown_mask = new_grown_mask
            iteration += 1
        
        return grown_mask
    
    def connected_component_expansion(self, mask, min_size=50):
        """
        Expand connected components and remove small artifacts
        """
        # Label connected components
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)
        
        # Create expanded mask
        expanded_mask = np.zeros_like(mask)
        
        for region in regions:
            if region.area >= min_size:
                # Get region coordinates
                coords = region.coords
                
                # Create convex hull for expansion
                try:
                    # Get bounding box with padding
                    min_row, min_col, max_row, max_col = region.bbox
                    padding = 3
                    
                    min_row = max(0, min_row - padding)
                    min_col = max(0, min_col - padding)
                    max_row = min(mask.shape[0], max_row + padding)
                    max_col = min(mask.shape[1], max_col + padding)
                    
                    # Fill expanded bounding box
                    for coord in coords:
                        y, x = coord
                        # Expand around each tumor pixel
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                                    expanded_mask[ny, nx] = 1
                
                except:
                    # Fallback: just copy original region
                    for coord in coords:
                        expanded_mask[coord[0], coord[1]] = 1
        
        return expanded_mask
    
    def tumor_boundary_smoothing(self, mask):
        """
        Smooth tumor boundaries while preserving expansion
        """
        # Apply Gaussian smoothing
        smoothed = ndimage.gaussian_filter(mask.astype(float), sigma=1.0)
        
        # Re-threshold with lower threshold to maintain expansion
        smoothed_mask = (smoothed >= 0.3).astype(np.uint8)
        
        # Fill small holes
        filled_mask = ndimage.binary_fill_holes(smoothed_mask).astype(np.uint8)
        
        return filled_mask
    
    def comprehensive_tumor_expansion(self, prediction):
        """
        Apply comprehensive tumor expansion pipeline
        """
        print("🔍 Starting comprehensive tumor expansion...")
        
        # Step 1: Aggressive threshold detection
        primary_mask, secondary_mask = self.aggressive_threshold_detection(
            prediction, self.expansion_params['detection_threshold']
        )
        print(f"   ✓ Primary detection: {np.sum(primary_mask)} pixels")
        print(f"   ✓ Secondary detection: {np.sum(secondary_mask)} pixels")
        
        # Step 2: Morphological expansion
        morpho_expanded = self.morphological_tumor_expansion(
            primary_mask, self.expansion_params['expansion_iterations']
        )
        print(f"   ✓ Morphological expansion: {np.sum(morpho_expanded)} pixels")
        
        # Step 3: Region growing expansion
        region_grown = self.region_growing_expansion(
            prediction, morpho_expanded, self.expansion_params['expansion_threshold']
        )
        print(f"   ✓ Region growing: {np.sum(region_grown)} pixels")
        
        # Step 4: Connected component expansion
        component_expanded = self.connected_component_expansion(
            region_grown, self.expansion_params['min_tumor_size']
        )
        print(f"   ✓ Component expansion: {np.sum(component_expanded)} pixels")
        
        # Step 5: Boundary smoothing
        final_mask = self.tumor_boundary_smoothing(component_expanded)
        print(f"   ✓ Final smoothed mask: {np.sum(final_mask)} pixels")
        
        expansion_ratio = np.sum(final_mask) / max(np.sum(primary_mask), 1)
        print(f"🎯 Total expansion ratio: {expansion_ratio:.2f}x")
        
        return {
            'original': primary_mask,
            'morphological': morpho_expanded,
            'region_grown': region_grown,
            'component_expanded': component_expanded,
            'final': final_mask,
            'expansion_ratio': expansion_ratio
        }
    
    def visualize_expansion_process(self, image, prediction, expansion_results):
        """
        Visualize the tumor expansion process
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Prediction heatmap
        axes[0, 1].imshow(prediction, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('Prediction Heatmap')
        axes[0, 1].axis('off')
        
        # Original detection
        axes[0, 2].imshow(image, cmap='gray', alpha=0.7)
        axes[0, 2].imshow(expansion_results['original'], cmap='Reds', alpha=0.5)
        axes[0, 2].set_title(f'Original Detection\n({np.sum(expansion_results["original"])} pixels)')
        axes[0, 2].axis('off')
        
        # Morphological expansion
        axes[0, 3].imshow(image, cmap='gray', alpha=0.7)
        axes[0, 3].imshow(expansion_results['morphological'], cmap='Oranges', alpha=0.5)
        axes[0, 3].set_title(f'Morphological Expansion\n({np.sum(expansion_results["morphological"])} pixels)')
        axes[0, 3].axis('off')
        
        # Region growing
        axes[1, 0].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 0].imshow(expansion_results['region_grown'], cmap='Yellows', alpha=0.5)
        axes[1, 0].set_title(f'Region Growing\n({np.sum(expansion_results["region_grown"])} pixels)')
        axes[1, 0].axis('off')
        
        # Component expansion
        axes[1, 1].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(expansion_results['component_expanded'], cmap='Greens', alpha=0.5)
        axes[1, 1].set_title(f'Component Expansion\n({np.sum(expansion_results["component_expanded"])} pixels)')
        axes[1, 1].axis('off')
        
        # Final result
        axes[1, 2].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 2].imshow(expansion_results['final'], cmap='Blues', alpha=0.5)
        axes[1, 2].set_title(f'Final Expanded Mask\n({np.sum(expansion_results["final"])} pixels)')
        axes[1, 2].axis('off')
        
        # Expansion comparison
        axes[1, 3].imshow(image, cmap='gray', alpha=0.5)
        axes[1, 3].imshow(expansion_results['original'], cmap='Reds', alpha=0.3, label='Original')
        axes[1, 3].imshow(expansion_results['final'], cmap='Blues', alpha=0.3, label='Expanded')
        axes[1, 3].set_title(f'Before vs After\nExpansion: {expansion_results["expansion_ratio"]:.2f}x')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('tumor_expansion_process.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def batch_process_predictions(self, images, predictions):
        """
        Process multiple predictions with tumor expansion
        """
        expanded_results = []
        
        print(f"🚀 Processing {len(predictions)} predictions with tumor expansion...")
        
        for i, (image, prediction) in enumerate(zip(images, predictions)):
            print(f"\n📊 Processing image {i+1}/{len(predictions)}...")
            
            # Apply comprehensive expansion
            expansion_result = self.comprehensive_tumor_expansion(prediction)
            
            # Store results
            expanded_results.append({
                'image_index': i,
                'original_tumor_pixels': np.sum(expansion_result['original']),
                'expanded_tumor_pixels': np.sum(expansion_result['final']),
                'expansion_ratio': expansion_result['expansion_ratio'],
                'expanded_mask': expansion_result['final']
            })
            
            # Visualize first few results
            if i < 3:
                self.visualize_expansion_process(image, prediction, expansion_result)
        
        return expanded_results

def main():
    """
    Main function to demonstrate tumor expansion post-processing
    """
    print("🎯 TUMOR EXPANSION POST-PROCESSING SYSTEM")
    print("=" * 55)
    print("🔥 MAXIMIZING TUMOR DETECTION COVERAGE")
    print("🎯 CAPTURING MORE TUMOR PARTS")
    print("⚡ EXPANDING TUMOR BOUNDARIES")
    print("=" * 55)
    
    # Initialize processor
    processor = TumorExpansionProcessor()
    
    print("📊 Usage Instructions:")
    print("=" * 30)
    print("1. Load your model predictions:")
    print("   predictions = model.predict(X_test)")
    print()
    print("2. Apply tumor expansion:")
    print("   expanded_results = processor.batch_process_predictions(X_test, predictions)")
    print()
    print("3. Use expanded masks for evaluation:")
    print("   expanded_masks = [result['expanded_mask'] for result in expanded_results]")
    print()
    
    print("🎯 Key Benefits:")
    print("✅ Captures weak tumor signals")
    print("✅ Expands tumor boundaries")
    print("✅ Fills gaps within tumors")
    print("✅ Removes small artifacts")
    print("✅ Smooths tumor boundaries")
    print("✅ Increases tumor detection sensitivity")
    print()
    
    print("📈 Expected Results:")
    print("🔹 1.5-3x increase in detected tumor pixels")
    print("🔹 Better tumor boundary definition")
    print("🔹 Reduced false negatives")
    print("🔹 More complete tumor segmentation")
    print()
    
    print("✅ TUMOR EXPANSION POST-PROCESSING READY!")
    print("🎯 Your predictions will now capture MORE TUMOR PARTS!")

if __name__ == "__main__":
    main()