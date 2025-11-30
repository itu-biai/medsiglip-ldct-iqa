# Prompt-Conditioned FiLM and Multi-Scale Fusion on MedSigLIP for Low-Dose CT Quality Assessment

This project implements a prompt-conditioned framework built on MedSigLIP that injects textual priors via Feature-wise Linear Modulation (FiLM) and multi-scale pooling. Text prompts condition patch-token features on clinical intent, enabling data-efficient learning. The architecture combines global, local, and texture-aware pooling through separate regression heads fused by a lightweight MLP, trained with pairwise ranking loss.

## Project Overview

The model predicts Mean Opinion Score (MOS) values (0-4 scale) for low-dose CT images, enabling automated quality assessment for medical imaging applications. Key innovations include:

1. **MedSigLIP Backbone** - Medical domain-adapted vision-language model
2. **FiLM Prompt Conditioning** - Text-guided feature modulation for task-specific adaptation
3. **Multi-Scale Feature Extraction** - Global, local, and texture-level quality assessment
4. **Pairwise Ranking Loss** - RankNet-based loss for better correlation metrics

## Directory Structure

```
.
├── config.py           # Configuration settings and hyperparameters
├── model.py            # Model architecture (PromptedMultiScaleMedSigLIPIQA)
├── train.py            # Training script with 5-fold cross-validation
├── test.py             # Evaluation script with PLCC, SROCC, KROCC metrics
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore file
└── assets/
    └── architecture.png
```

## Key Features

- **MedSigLIP Integration**: Leverages Google's medical domain SigLIP model (`google/medsiglip-448`)
- **FiLM Conditioning**: Text prompts guide the model to focus on diagnostic quality aspects
- **Multi-Scale Pooling**: 
  - Global average pooling for overall quality
  - Local average pooling (4 regions) for regional quality
  - Texture max pooling (2 regions) for texture/detail quality
- **Advanced Training**:
  - Mixed precision training (AMP) for faster training
  - Gradient accumulation for effective larger batch sizes
  - Cosine annealing learning rate scheduler
  - Early stopping with patience
- **5-Fold Cross-Validation**: Robust model evaluation and selection
- **Comprehensive Metrics**: PLCC, SROCC, KROCC correlation coefficients

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/itu-biai/medsiglip_ldct_iqa.git
   cd medsiglip_ldct_iqa
   ```

2. Create a conda environment and install dependencies:
   ```bash
   conda create -n medsiglip_ldct_iqa python=3.10
   conda activate medsiglip_ldct_iqa
   pip install -r requirements.txt
   ```

## Requirements

The project dependencies are listed in `requirements.txt`:
- PyTorch (≥2.0.0) and torchvision (≥0.15.0) for deep learning
- Transformers (≥4.30.0) for MedSigLIP model
- NumPy (≥1.24.0) for numerical computations
- Pandas (≥2.0.0) for data handling
- SciPy (≥1.10.0) for statistical metrics
- scikit-learn (≥1.2.0) for cross-validation and metrics
- imageio (≥2.31.0) and Pillow (≥9.5.0) for image processing
- pydicom (≥2.4.0) for DICOM file support
- tqdm (≥4.65.0) for progress bars

## Configuration

All configuration settings are centralized in `config.py`. Key parameters to adjust:

### File Paths
```python
"checkpoint_path": "path/to/save/medsiglip_iqa"      # Model checkpoint save location
"json_path": "path/to/train.json"            # Training data annotations
"data_dir": "path/to/train/images"           # Training images directory
"test_json_path": "path/to/test.json"        # Test data annotations
"test_data_dir": "path/to/test/images"       # Test images directory
```

### Training Parameters
```python
"learning_rate": 1e-5      # Learning rate
"batch_size": 4            # Batch size per GPU
"num_epochs": 22           # Maximum training epochs
"weight_decay": 1e-4       # AdamW weight decay
"grad_accum_steps": 2      # Gradient accumulation steps
"use_amp": True            # Enable mixed precision training
"T_max": 100                # Cosine annealing T_max (usually = num_epochs)
"eta_min": 1e-7            # Cosine annealing minimum learning rate
```

### Model Parameters
```python
"medsiglip_model": "google/medsiglip-448"  # Backbone model
"image_size": 448                           # Input image resolution
"use_text_prompt": True                     # Enable FiLM conditioning
"film_strength": 1.0                        # FiLM modulation strength (0=off)
"sigmoid_tau": 2.0                          # Output sigmoid temperature
"bound_output_0_4": True                    # Constrain output to [0,4]
```

## Dataset

The model was trained and evaluated on the LDCTIQA2023 Challenge dataset:

- **Training Data:** [Zenodo - LDCT-IQA 2023](https://zenodo.org/records/7833096#.ZEFywOxBzn5)
- **Test Data:** [Google Drive](https://drive.google.com/file/d/1DbZfTj4it-3QRTuJO73r1iQyHLRwPS6H/view)

## Data Format

The training and test data should be organized as:

```
data_dir/
├── image1.png
├── image2.png
└── ...

train.json / test.json:
{
    "image1.png": 3.5,
    "image2.png": 2.1,
    ...
}
```

- **Images**: PNG, JPG, TIFF, or DICOM formats supported
- **Annotations**: JSON file mapping filename to MOS score (0-4 scale)

## Usage

### Training

```bash
# Train with 5-fold cross-validation
python train.py
```

The training script will:
- Load and preprocess the dataset
- Perform 5-fold cross-validation
- Save the best model checkpoint based on validation loss for each fold
- Copy the best fold model as `medsiglip_iqa_best.pth`
- Log training/validation metrics (loss, MAE, RMSE, Pearson correlation) per epoch

### Testing

```bash
# Evaluate on test set
python test.py
```

Before running, update `config.py`:
```python
"checkpoint_path": "path/to/medsiglip_iqa_best.pth"
```

The test script outputs:
- **PLCC**: Pearson Linear Correlation Coefficient
- **SROCC**: Spearman Rank Order Correlation Coefficient  
- **KROCC**: Kendall Rank Order Correlation Coefficient

## Model Architecture

### PromptedMultiScaleMedSigLIPIQA

![Model Architecture](assets/architecture.png)

*Figure: Prompt-conditioned FiLM with multi-scale (global/local/texture) pooling.*

### Loss Function (Pairwise Ranking Loss)

We use pure pairwise ranking loss following RankNet with temperature $\tau_{\mathrm{rank}}=0.5$:

$$\mathcal{L}_{\mathrm{rank}} = \frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}} \log\left(1+e^{\frac{-s_{ij}(\hat y_i-\hat y_j)}{\tau_{\mathrm{rank}}}}\right)$$

where $\mathcal{P}$ is the set of non-tied pairs and $s_{ij}=\mathrm{sign}(y_i-y_j)$.

## Evaluation Metrics

The model is evaluated using standard IQA correlation metrics:
- **PLCC (Pearson Linear Correlation Coefficient)**: Measures linear correlation
- **SROCC (Spearman Rank Order Correlation Coefficient)**: Measures monotonic relationship
- **KROCC (Kendall Rank Order Correlation Coefficient)**: Measures ordinal association

Higher values (closer to 1.0) indicate better performance.

## Pre-trained Weights

Pre-trained model weights are available on Google Drive:
- 📥 **[Download best_model.pth](https://drive.google.com/drive/folders/1ShbXiggqHdVGdCkYe8IK2RKP6vErSogV?usp=sharing)**

**Test Results:**
| PLCC | SROCC | KROCC | OVERALL |
|------|-------|-------|---------|
| 0.9575 | 0.9561 | 0.8301 | 2.7436 |

To use the pre-trained model:
1. Download `best_model.pth` from the link above
2. Place it in the project root directory
3. Set in `config.py`:
```python
"checkpoint_path": "best_model.pth"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please consider citing:

```bibtex
@inproceedings{to_be_filled,
  title={Prompt-Conditioned FiLM and Multi-Scale Fusion on MedSigLIP for Low-Dose CT Quality Assessment},
  author={To be filled},
  booktitle={To be filled},
  year={2025},
  url={To be filled}
}
```

## Acknowledgements

- [Google MedSigLIP](https://huggingface.co/google/medsiglip-448) for the pretrained medical vision-language model
- [LDCT-IQA Challenge](https://ldctiqac2023.grand-challenge.org/) for the dataset

## Contact

For questions and feedback, please open an issue in the GitHub repository.
