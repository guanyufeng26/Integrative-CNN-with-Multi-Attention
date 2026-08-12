# Integrative-CNN-with-Multi-Attention
## Project Overview
This open-source repository contains all the Python implementation codes of our protein adhesion binary prediction model.
We design an integrative CNN model fused with three attention modules: CBAM, SENet and global self-attention.
The model takes amino acid sequence features as input, and completes binary classification to judge whether a protein has adhesion property.

### Key Input Features
We adopt three types of protein sequence features, detailed as follows:

| Feature Name | Dimension |
|--------------|-----------|
| AA Pairs     | 6D        |
| PC properties| 68D       |
| ProtT5 embedding | 256D |
