# Zero-Shot Cross-Lingual Natural Language Inference with Bi-Encoders and Cross-Encoders

## Overview

This project investigates and compares the zero-shot cross-lingual transfer capabilities of bi-encoder and cross-encoder architectures for the task of Natural Language Inference (NLI). The models are trained on the English Multi-Genre NLI (MNLI) dataset and evaluated on the Spanish and German test sets of the Cross-lingual NLI corpus (XNLI) without any fine-tuning on these target languages.

The goal is to understand the effectiveness of these two different model architectures in transferring NLI understanding across languages in a zero-shot setting.

## Project Files

- `zero-shot-xnli-cross-bi.ipynb`: The main Jupyter Notebook containing the full pipeline for data loading, preprocessing, model training, evaluation, and analysis.
- `requirements.txt`: A list of all required Python libraries and their versions to run the notebook.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd zero-shot-xnli-cross-bi-test

2.  **Install required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Datasets:** The notebook automatically downloads the MNLI and XNLI datasets using the `datasets` library. Ensure your environment has internet access when running the notebook.

3.  **Verify GPU Availability:** The notebook is optimized for GPU usage. Ensure that PyTorch is installed with CUDA support and that a compatible GPU is available.

## Usage

To run the project, open and execute the Jupyter Notebook (zero-shot-xnli-cross-bi.ipynb) sequentially.
The notebook performs the following steps:

1.  **Install Required Libraries:** Ensures all necessary Python libraries are installed.
2. **Load Datasets:** Downloads and loads the MNLI and XNLI datasets.
3. **Data Inspection:** Analyzes class distributions and token lengths in the datasets.
4. **Preprocessing:** Tokenizes and formats the datasets for both bi-encoder and cross-encoder models.
5. **Train Bi-Encoder:** Trains a bi-encoder model on the MNLI training set.
6. **Evaluate Bi-Encoder:** Evaluates the bi-encoder on the MNLI validation set and the Spanish and German XNLI test sets.
7. **Retrain Cross-Encoder:** Retrains a cross-encoder model on the full MNLI dataset.
8. **Evaluate Cross-Encoder:** Evaluates the cross-encoder on the MNLI validation set and the Spanish and German XNLI test sets.
9. **Statistical Significance Testing:** Performs bootstrap tests to assess the statistical significance of accuracy differences between the models.
10. **Visualization:** Generates bar charts comparing accuracy and F1-scores, as well as confusion matrix heatmaps.

## Results

The notebook outputs the following results:

- **Evaluation Metrics**: Loss, accuracy, precision, recall, F1-score, and per-class metrics for both models on all datasets.
- **Visualizations**:
  - Bar charts comparing accuracy and macro F1-scores of the bi-encoder and cross-encoder on the MNLI validation and XNLI test sets.
  - Confusion matrix heatmaps for each model on each dataset.
- **Statistical Significance**:
  - Results of bootstrap tests to assess whether the observed differences in accuracy between the models are statistically significant.
- **Saved Models**:
  - The retrained cross-encoder model is saved in the `cross_encoder_retrained_final/` directory.

## Key Findings

- **Bi-Encoder Performance**:
  - Achieved 67.59% accuracy and 67.53% F1-score on the MNLI validation set.
  - Demonstrated zero-shot performance on Spanish XNLI with 60.46% accuracy and 60.47% F1-score, and on German XNLI with 57.25% accuracy and 57.29% F1-score.

- **Cross-Encoder Performance**:
  - Significantly outperformed the bi-encoder on the MNLI validation set with 82.63% accuracy and 82.59% F1-score.
  - Showed superior zero-shot transfer to Spanish XNLI (74.87% accuracy, 74.89% F1-score) and German XNLI (72.02% accuracy, 72.03% F1-score).

- **Performance Comparison (Zero-Shot)**:
  - The cross-encoder exhibited a substantial performance advantage over the bi-encoder in the zero-shot setting on both Spanish and German NLI tasks. This suggests the joint encoding strategy is more effective for cross-lingual transfer in this context, with accuracy improvements of approximately 15% on both target languages.

- **Statistical Significance**:
  - While the observed performance differences were notable (around 12-15% in accuracy on the zero-shot tasks), bootstrap tests for accuracy did not yield p-values below the 0.05 threshold (MNLI: p=0.5048, Spanish: p=0.5057, German: p=0.5013). 
  - This indicates that the observed differences are not statistically significant based on this analysis. However, the consistent and practically meaningful performance gap suggests further investigation is warranted, potentially with larger sample sizes or alternative statistical methods.

## Further Work

- Explore different pre-trained multilingual models as the base for the bi-encoder and cross-encoder.
- Experiment with different training strategies or hyperparameters.
- Investigate the impact of different alignment techniques for bi-encoders to improve cross-lingual transfer.
- Evaluate the models on a wider range of languages in the XNLI dataset.
- Analyze the types of errors made by each model through the confusion matrices to gain deeper insights into their cross-lingual understanding.
