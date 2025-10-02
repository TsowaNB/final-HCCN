"""
Project Flow Chart Generator
Creates a comprehensive flow chart for the Hybrid U-Net Glioma Segmentation Project
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_project_flowchart():
    """Create a comprehensive project flow chart."""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 25)
    ax.axis('off')
    
    # Define colors
    colors = {
        'data': '#E3F2FD',      # Light Blue
        'process': '#F3E5F5',   # Light Purple
        'model': '#E8F5E8',     # Light Green
        'quantum': '#FFF3E0',   # Light Orange
        'evaluation': '#FFEBEE', # Light Red
        'output': '#F1F8E9'     # Light Lime
    }
    
    # Title
    ax.text(5, 24, 'Hybrid U-Net for Lower-grade Glioma Segmentation', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(5, 23.5, 'Complete Project Workflow', 
            fontsize=14, ha='center', style='italic')
    
    # Step 1: Data Loading and Visualization
    step1_box = FancyBboxPatch((0.5, 21), 4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['data'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(2.5, 21.75, 'STEP 1: Data Loading & Visualization', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(2.5, 21.4, '• Load Kaggle 3M LGG Dataset', fontsize=10, ha='center')
    ax.text(2.5, 21.1, '• Dataset Statistics & Overview', fontsize=10, ha='center')
    
    # Step 2: Data Preprocessing
    step2_box = FancyBboxPatch((5.5, 21), 4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['process'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(step2_box)
    ax.text(7.5, 21.75, 'STEP 2: Data Preprocessing', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(7.5, 21.4, '• Normalization & Augmentation', fontsize=10, ha='center')
    ax.text(7.5, 21.1, '• Train/Val/Test Split', fontsize=10, ha='center')
    
    # Data Flow Arrow
    arrow1 = ConnectionPatch((2.5, 21), (2.5, 19.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc="black")
    ax.add_patch(arrow1)
    
    # Preprocessing Details
    preprocess_box = FancyBboxPatch((0.5, 18), 4, 1.2, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['process'], 
                                   edgecolor='gray', linewidth=1)
    ax.add_patch(preprocess_box)
    ax.text(2.5, 18.8, 'Preprocessing Pipeline', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.5, 18.5, '• Z-score/Min-Max Normalization', fontsize=9, ha='center')
    ax.text(2.5, 18.3, '• Albumentations Augmentation', fontsize=9, ha='center')
    ax.text(2.5, 18.1, '• Data Generators', fontsize=9, ha='center')
    
    # Dataset Split
    split_box = FancyBboxPatch((5.5, 18), 4, 1.2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['data'], 
                              edgecolor='gray', linewidth=1)
    ax.add_patch(split_box)
    ax.text(7.5, 18.8, 'Dataset Splits', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.5, 18.5, '• Training: 70%', fontsize=9, ha='center')
    ax.text(7.5, 18.3, '• Validation: 15%', fontsize=9, ha='center')
    ax.text(7.5, 18.1, '• Testing: 15%', fontsize=9, ha='center')
    
    # Step 3: Model Architectures
    ax.text(5, 16.5, 'STEP 3: Model Architectures', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Classical U-Net
    classical_box = FancyBboxPatch((0.5, 14.5), 4, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['model'], 
                                  edgecolor='blue', linewidth=2)
    ax.add_patch(classical_box)
    ax.text(2.5, 15.4, 'Classical U-Net', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.5, 15.1, '• Encoder-Decoder Architecture', fontsize=9, ha='center')
    ax.text(2.5, 14.9, '• Skip Connections', fontsize=9, ha='center')
    ax.text(2.5, 14.7, '• Batch Normalization', fontsize=9, ha='center')
    
    # Hybrid Quantum U-Net
    hybrid_box = FancyBboxPatch((5.5, 14.5), 4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['quantum'], 
                               edgecolor='orange', linewidth=2)
    ax.add_patch(hybrid_box)
    ax.text(7.5, 15.4, 'Hybrid Quantum U-Net', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.5, 15.1, '• Classical + Quantum Layers', fontsize=9, ha='center')
    ax.text(7.5, 14.9, '• Variational Quantum Circuits', fontsize=9, ha='center')
    ax.text(7.5, 14.7, '• PennyLane Integration', fontsize=9, ha='center')
    
    # Quantum Circuit Details
    quantum_detail = FancyBboxPatch((6, 13), 3, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['quantum'], 
                                   edgecolor='orange', linewidth=1, alpha=0.7)
    ax.add_patch(quantum_detail)
    ax.text(7.5, 13.6, 'Quantum Circuit', fontsize=10, fontweight='bold', ha='center')
    ax.text(7.5, 13.4, '• 4 Qubits', fontsize=9, ha='center')
    ax.text(7.5, 13.2, '• 2 Variational Layers', fontsize=9, ha='center')
    
    # Step 4: Training Pipeline
    training_box = FancyBboxPatch((2, 11), 6, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['process'], 
                                 edgecolor='purple', linewidth=2)
    ax.add_patch(training_box)
    ax.text(5, 11.9, 'STEP 4: Training Pipeline', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 11.6, '• Combined Loss (Dice + Focal + BCE)', fontsize=10, ha='center')
    ax.text(5, 11.4, '• Adam Optimizer with Callbacks', fontsize=10, ha='center')
    ax.text(5, 11.2, '• TensorBoard Logging', fontsize=10, ha='center')
    
    # Training Components
    loss_box = FancyBboxPatch((0.5, 9.5), 2.5, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['process'], 
                             edgecolor='gray', linewidth=1)
    ax.add_patch(loss_box)
    ax.text(1.75, 10.1, 'Loss Functions', fontsize=10, fontweight='bold', ha='center')
    ax.text(1.75, 9.8, '• Dice Loss', fontsize=9, ha='center')
    ax.text(1.75, 9.6, '• Focal Loss', fontsize=9, ha='center')
    
    callbacks_box = FancyBboxPatch((3.5, 9.5), 3, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['process'], 
                                  edgecolor='gray', linewidth=1)
    ax.add_patch(callbacks_box)
    ax.text(5, 10.1, 'Callbacks', fontsize=10, fontweight='bold', ha='center')
    ax.text(5, 9.8, '• Early Stopping • Model Checkpoint', fontsize=9, ha='center')
    ax.text(5, 9.6, '• Learning Rate Reduction', fontsize=9, ha='center')
    
    monitoring_box = FancyBboxPatch((7, 9.5), 2.5, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['process'], 
                                   edgecolor='gray', linewidth=1)
    ax.add_patch(monitoring_box)
    ax.text(8.25, 10.1, 'Monitoring', fontsize=10, fontweight='bold', ha='center')
    ax.text(8.25, 9.8, '• TensorBoard', fontsize=9, ha='center')
    ax.text(8.25, 9.6, '• CSV Logging', fontsize=9, ha='center')
    
    # Step 5: Evaluation
    eval_box = FancyBboxPatch((2, 7.5), 6, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['evaluation'], 
                             edgecolor='red', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(5, 8.4, 'STEP 5: Model Evaluation & Comparison', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 8.1, '• Comprehensive Metrics Calculation', fontsize=10, ha='center')
    ax.text(5, 7.9, '• Statistical Analysis & Visualization', fontsize=10, ha='center')
    ax.text(5, 7.7, '• Performance Comparison Report', fontsize=10, ha='center')
    
    # Evaluation Metrics
    metrics_box = FancyBboxPatch((0.5, 5.5), 4.5, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['evaluation'], 
                                edgecolor='gray', linewidth=1)
    ax.add_patch(metrics_box)
    ax.text(2.75, 6.5, 'Evaluation Metrics', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.75, 6.2, '• Dice Coefficient • IoU Score', fontsize=9, ha='center')
    ax.text(2.75, 6.0, '• Precision • Recall • F1 Score', fontsize=9, ha='center')
    ax.text(2.75, 5.8, '• Hausdorff Distance', fontsize=9, ha='center')
    
    # Outputs
    output_box = FancyBboxPatch((5.5, 5.5), 4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='green', linewidth=1)
    ax.add_patch(output_box)
    ax.text(7.5, 6.5, 'Project Outputs', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.5, 6.2, '• Trained Models • Evaluation Report', fontsize=9, ha='center')
    ax.text(7.5, 6.0, '• Comparison Visualizations', fontsize=9, ha='center')
    ax.text(7.5, 5.8, '• Performance Metrics', fontsize=9, ha='center')
    
    # Final Results
    results_box = FancyBboxPatch((2, 3.5), 6, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='green', linewidth=2)
    ax.add_patch(results_box)
    ax.text(5, 4.4, 'FINAL DELIVERABLES', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 4.1, '• Classical vs Hybrid Model Comparison', fontsize=10, ha='center')
    ax.text(5, 3.9, '• Clinical Application Ready Models', fontsize=10, ha='center')
    ax.text(5, 3.7, '• Comprehensive Documentation', fontsize=10, ha='center')
    
    # Future Work
    future_box = FancyBboxPatch((1, 1.5), 8, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F5F5F5', 
                               edgecolor='gray', linewidth=1, linestyle='--')
    ax.add_patch(future_box)
    ax.text(5, 2.4, 'FUTURE EXTENSIONS', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 2.1, '• Streamlit Web Application for Real-time Segmentation', fontsize=10, ha='center')
    ax.text(5, 1.9, '• Clinical Deployment and Validation', fontsize=10, ha='center')
    ax.text(5, 1.7, '• Extended Quantum Circuit Architectures', fontsize=10, ha='center')
    
    # Add flow arrows
    arrows = [
        # Main flow
        ((5, 21), (5, 16.8)),  # Data to Models
        ((5, 14.2), (5, 12.5)), # Models to Training
        ((5, 11), (5, 9)),      # Training to Evaluation
        ((5, 7.5), (5, 5)),     # Evaluation to Results
        ((5, 3.5), (5, 3)),     # Results to Future
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=20, fc="black", linewidth=2)
        ax.add_patch(arrow)
    
    # Add side arrows for parallel processes
    side_arrows = [
        ((4.5, 21.75), (5.5, 21.75)),  # Data loading to preprocessing
        ((2.5, 18), (2.5, 16)),        # Preprocessing to models
        ((7.5, 18), (7.5, 16)),        # Split to models
    ]
    
    for start, end in side_arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=15, fc="gray", alpha=0.7)
        ax.add_patch(arrow)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['data'], label='Data Processing'),
        patches.Patch(color=colors['process'], label='Processing Pipeline'),
        patches.Patch(color=colors['model'], label='Classical Model'),
        patches.Patch(color=colors['quantum'], label='Quantum Model'),
        patches.Patch(color=colors['evaluation'], label='Evaluation'),
        patches.Patch(color=colors['output'], label='Outputs')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('hybrid_unet_project_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Project flow chart created successfully!")
    print("📁 Saved as: hybrid_unet_project_flowchart.png")

if __name__ == "__main__":
    create_project_flowchart()