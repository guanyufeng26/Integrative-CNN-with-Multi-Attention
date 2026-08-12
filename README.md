# 🧬Integrative-CNN-with-Multi-Attention

---

## 🔗Project Overview
This open-source repository contains all the Python implementation codes of our protein adhesion binary prediction model.
We design an integrative CNN model fused with three attention modules: SENet, CBAM and global self-attention.
The model takes amino acid sequence features as input, and completes binary classification to judge whether a protein has adhesion property.
### Input Features
| Feature  | Dimension |
|--------------|-----------|
| AA Pairs     | 6D        |
| PC properties| 68D       |
| ProtT5 embedding | 256D |

---

## ⚙️Environment Requirements
| Library  | Version |
|----------|---------|
| python   | 3.8.18  |
| numpy    | 1.24.4  |
| pandas   | 2.0.3   |
| pillow   | 10.4.0  |
| torch    | 2.4.1   |
| catboost | 1.2.8   |
| function | 1.2.0   |
| pycaret  | 3.2.0   |
| xgboost  | 2.1.4   |
### Additional required packages for protein sequence processing & visualization
pip install biopython transformers umap-learn matplotlib seaborn scikit-learn tqdm

---

## 📁Directory Structure

| Folder/File | Description |
|-------------|-------------|
| ablation_train/ | Stores all codes for ablation experiments of different attention modules. |
| data/ | Contains all datasets used in the experiments. |
| feature/ | Extract three groups of protein features.|
| UMAP_draw.py | UMAP dimensionality reduction & feature visualization script. |
| case_predict.py | External real protein case prediction. |
| test.py | Model evaluation. |


## 🧪 Experimental Procedures
### Step 1 — Data Preprocessing & Dataset Partition
• Collect raw protein data, remove redundant sequences and filter samples by sequence length.
• Randomly split the processed dataset into training set, validation set and independent test set at an 8:1:1 ratio.

### Step 2 — Feature Extraction
• Execute scripts under the `feature/` folder to extract three groups of protein features: 
| Feature | Description |
|---------|-------------|
| `6D AA Pairs` | Amino acid pair based on physicochemical properties |
| `68D PC properties` | iLearnPlus-based feature extraction |
| `256D ProtT5 embedding` | ProtT5 embedding features |

### Step 3 — Model Training & Attention Ablation Experiments
• Train models with codes inside ablation_train/ to compare CNN combined with different attention mechanisms:
| Model |
|-------|
| CNN+SENet |
| CNN+CBAM |
| CNN+global self-attention |
| CNN+SENet+CBAM |
| CNN+SENet+global self-attention |
| CNN+CBAM+global self-attention |
| CNN+SENet+CBAM+global self-attention |

### Step 4 — Model Performance Evaluation
• Load the trained weight and run `test.py` on unseen independent test set.
• Automatically calculate classification metrics: AUC, Accuracy, Precision, Recall and F1-Score.

### Step 5 — Visualization & External Sample Prediction
• Execute `UMAP_draw.py` to implement UMAP dimensionality reduction and plot feature distribution.
• Run `case_predict.py` to complete inference on external real-world protein samples.
