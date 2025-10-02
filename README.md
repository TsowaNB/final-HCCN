# Hybrid U-Net Model for Lower-grade Glioma Segmentation

## 🧠 Project Overview

This project implements a hybrid U-Net model that combines classical convolutional layers with variational quantum circuits for accurate segmentation of lower-grade glioma in MRI scans. The project addresses the challenge of medical image segmentation by leveraging both classical deep learning and quantum computing approaches.

## 🎯 Objectives

1. **Design and implement** a hybrid U-Net model integrating classical convolutional layers with variational quantum circuits
2. **Preprocess and prepare** MRI datasets for effective training, validation, and testing
3. **Train, optimize, and evaluate** the hybrid model compared to classical segmentation models
4. **Develop metrics-based comparison** using Dice score, IoU, precision, and recall
5. **Deploy a Streamlit web application** for real-time MRI upload and tumor segmentation visualization

## 📁 Project Structure

```
final HCCN/
├── step1_data_loading.py      # Data loading and visualization
├── step2_preprocessing.py     # Data preprocessing and augmentation
├── step3_models.py           # Classical and Hybrid U-Net architectures
├── step4_training.py         # Training pipeline with custom losses
├── step5_evaluation.py       # Comprehensive model evaluation
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Kaggle account for dataset access

### Installation

1. **Clone or download** the project files to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle dataset**:
   - Download the LGG MRI Segmentation dataset from Kaggle
   - Extract to `/kaggle/input/lgg-mri-segmentation/kaggle_3m`
   - Or update the `DATASET_PATH` variable in each script

## 📊 Usage

### Step 1: Data Loading and Visualization
```python
python step1_data_loading.py
```
- Loads and visualizes random MRI samples
- Provides dataset statistics and overview
- Creates professional visualizations with tumor overlays

### Step 2: Data Preprocessing
```python
python step2_preprocessing.py
```
- Implements advanced preprocessing pipeline
- Applies normalization and augmentation techniques
- Prepares train/validation/test splits

### Step 3: Model Architectures
```python
python step3_models.py
```
- Creates Classical U-Net architecture
- Implements Hybrid Quantum U-Net with variational circuits
- Displays model summaries and parameter counts

### Step 4: Training Pipeline
```python
python step4_training.py
```
- Trains both classical and hybrid models
- Uses custom loss functions (Dice + Focal + BCE)
- Implements comprehensive callbacks and monitoring
- Saves training history and checkpoints

### Step 5: Model Evaluation
```python
python step5_evaluation.py
```
- Evaluates models using multiple metrics
- Generates comparison visualizations
- Creates comprehensive evaluation report
- Saves results and predictions

## 🔬 Technical Features

### Classical U-Net
- Standard encoder-decoder architecture
- Batch normalization and dropout
- Skip connections for feature preservation
- Optimized for medical image segmentation

### Hybrid Quantum U-Net
- Integrates variational quantum circuits
- Quantum-enhanced convolutional blocks
- PennyLane-based quantum layers
- Residual connections between classical and quantum features

### Advanced Preprocessing
- Multi-method normalization (Z-score, Min-Max, Percentile)
- Comprehensive augmentation pipeline
- Professional data generators
- Memory-efficient batch processing

### Training Pipeline
- Custom combined loss function
- Advanced callbacks (EarlyStopping, ReduceLROnPlateau)
- TensorBoard logging
- Comprehensive metrics tracking

### Evaluation Metrics
- Dice Coefficient
- Intersection over Union (IoU)
- Precision and Recall
- F1 Score
- Hausdorff Distance
- Statistical analysis and visualization

## 📈 Expected Results

The hybrid quantum model is expected to show:
- **Improved segmentation accuracy** in challenging cases
- **Better feature extraction** through quantum enhancement
- **Competitive performance** with classical approaches
- **Novel insights** into quantum-classical hybrid architectures

## 🛠️ Customization

### Model Parameters
- Adjust `filters` list in model constructors
- Modify `quantum_layers` indices for hybrid model
- Change `n_qubits` and `n_layers` for quantum circuits

### Training Configuration
- Modify `batch_size`, `epochs`, and learning rates
- Adjust loss function weights in `CombinedLoss`
- Customize augmentation pipeline parameters

### Evaluation Settings
- Change threshold values for binary classification
- Add custom metrics to evaluation pipeline
- Modify visualization parameters

## 🔧 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or image resolution
2. **Quantum Circuit Errors**: Ensure PennyLane is properly installed
3. **CUDA Issues**: Verify TensorFlow-GPU installation
4. **Dataset Path**: Update `DATASET_PATH` variables as needed

### Performance Optimization

- Use mixed precision training for faster computation
- Implement gradient checkpointing for memory efficiency
- Consider distributed training for large datasets
- Optimize quantum circuit depth for better performance

## 📚 References

- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Variational Quantum Circuits for Machine Learning
- PennyLane: Automatic differentiation of hybrid quantum-classical computations
- Medical Image Segmentation Techniques and Applications

## 🤝 Contributing

This project is designed for research and educational purposes. Feel free to:
- Experiment with different quantum circuit architectures
- Implement additional evaluation metrics
- Optimize the hybrid model design
- Extend to other medical imaging tasks

## 📄 License

This project is intended for academic and research purposes. Please cite appropriately if used in publications.

---

**Note**: This implementation is designed for the Kaggle environment with GPU support. Adjust paths and configurations as needed for your specific setup.