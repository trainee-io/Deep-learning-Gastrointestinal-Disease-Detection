# Gastrointestinal Disease Detection using Deep Learning

A custom CNN trained from scratch to classify gastrointestinal diseases from endoscopy images, paired with a statistical co-occurrence analysis to uncover relationships between diseases.

## Key Results
- Built a **custom 5-block CNN** (no pretrained weights) achieving **92% training accuracy** and **64% test accuracy** across 23 disease categories
- Per-class **AUC scores above 0.95** for most disease categories, showing strong class separation despite dataset imbalance
- Identified a clinically plausible **0.35 correlation** between two disease labels (A1–G1) through co-occurrence analysis
- Built a **logistic regression baseline** for binary disease detection, reaching an **AUC of 0.90** for pre-pyloric ulcer detection, providing a useful contrast to the CNN

## Methods
1. **Data Pipeline** — custom Keras Sequence generator; images resized to 224x224, normalized, with light augmentation (horizontal flips) applied only to training data
2. **CNN Architecture** — 5 convolutional blocks (32 → 512 filters) with L2 regularization and Dropout to control overfitting; trained with Adam optimizer and early stopping
3. **Evaluation** — accuracy, precision, recall, F1-score, Matthews Correlation Coefficient (MCC), and per-class AUC (One-vs-Rest)
4. **Co-occurrence & Correlation Analysis** — aggregated CNN probability outputs across the test set to build a disease–disease similarity matrix
5. **Logistic Regression Baseline** — binary classifier on handcrafted/CNN-derived features, evaluated separately for comparison against the CNN

## Results

### Training vs Validation Performance
Training accuracy reached 92%, while validation plateaued around 64%, showing a generalization gap that a single accuracy number alone would hide.

### Confusion Matrix
Most disease categories were well separated, with some confusion between visually similar categories (e.g., L4 and L5).

### AUC Per Class
Nearly all classes scored above 0.95 AUC, confirming strong discriminative ability even where raw accuracy was lower.

## Repository Structure
```
├── GI_Disease_Detection.ipynb        → Full training & analysis notebook
└── GI_Disease_Detection_Report.pdf   → Full project report
```

## Full Report
See [GI_Disease_Detection_Report.pdf](GI_Disease_Detection_Report.pdf) for complete methodology, dataset details, and findings.

## Limitations
- Dataset diversity and class imbalance affected performance on rarer categories
- The HuggingFace-style generalization gap (92% train vs 64% test) reflects a common deep learning failure mode on limited medical imaging data
- Correlation analysis reflects co-occurrence patterns in this dataset only and is not a clinical diagnostic tool
