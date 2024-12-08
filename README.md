# Plant Disease Identification with Dual-Branch Deep Learning

## Project Overview
This repository contains an advanced deep learning framework for plant disease identification, utilizing a dual-branch architecture that leverages both local and global features for improved classification accuracy. The project addresses the challenge of unseen crop-disease pair composition scenarios and extends to out-of-distribution capabilities across laboratory and field conditions.

## Models
The repository contains three main models:

1. **Base Crop Classifier**: 
   - Located in `/Base-Model/Crop/`
   - Specialized for crop classification
   - Pre-trained model available as `best_classifier.pth`

2. **Base Disease Classifier**:
   - Located in `/Base-Model/Disease/`
   - Focused on disease identification
   - Pre-trained model available as `disease_classifier.pth`

3. **Compositional Contrastive Learning Model (CCLM)**:
   - Main integrated model combining crop and disease classification
   - Implementation in `combined_model_6_54_t_a.py`
   - Features dual-branch architecture with attention mechanisms
   - Pre-trained model available as `best_model.pth`

## Architecture Highlights

The CCLM architecture utilizes a dual-branch approach to improve classification accuracy:
- Local branch focuses on disease-specific features
- Global branch captures broader contextual information for crop classification
- Integrates attention mechanisms for enhanced feature extraction
- Employs supervised contrastive learning for robust feature representation

## Data

### Datasets Used:
- PlantVillage: 49,489 laboratory images (38,994 training, 10,495 testing)
- PlantDoc: 71 field-condition images for real-world testing
- Coverage: 14 crop categories and 21 disease categories

### Data Preparation:
- Structured sampling with balanced representation
- Comprehensive augmentation techniques
- Organized in `/Data/` directory with preprocessing scripts

## Results & Evaluation

Evaluation metrics available in `/Evaluations/`:
- Confusion matrices
- Top-N accuracy measurements
- Cross-dataset generalization performance
- Detailed results in `evaluation_results.csv`

### Key Performance Metrics:
- PV Test Seen Accuracy:
  - Crop Top-1: 99.21%
  - Disease Top-1: 92.10%
- PV Test Unseen Accuracy:
  - Crop Top-1: 96.30%
  - Disease Top-1: 71.30%

## Interactive Demo

Try our model through our Hugging Face Space:
[Crop-Disease Classifier Demo](https://huggingface.co/spaces/MuniebAbdelrahman/Crop-Disease-Classifier)

## Training Results

Training visualizations and metrics available in `/Training_result/`:
- Loss curves
- Accuracy plots
- Model convergence analysis

## Team Members

- **Abdullahi** - Team Lead
- **Munieb** - Team Member
- **Yassir** - Team Member

