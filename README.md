# Persian Name Gender Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pFvKZ2-Wl6fdecF2bR0eLEog8Uos49fa#scrollTo=d4G88pI_1rmr)

This project is focused on predicting gender from Persian names using various machine learning and deep learning models. The project includes data cleaning, exploratory data analysis (EDA), model training, and evaluation. Given the sensitive nature of gender prediction, especially for public services, we aim for a high level of accuracy and reliability.

## Table of Contents
- [Introduction](#introduction)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Development](#model-development)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [Contact](#Contact)

## Introduction

Gender prediction based on Persian names can be challenging due to the nuanced and sometimes ambiguous nature of certain names. This project tackles this challenge by leveraging character-level machine learning and deep learning models. The final model is intended to be deployed as a public-facing service, where accuracy and reliability are critical.

## Data Collection and Preprocessing

The dataset consists of 6,175 Persian names along with their associated gender and usage frequency. Data preprocessing steps include:

- **Normalization:** Normalizing names while preserving key gender-specific characters to ensure that important distinctions in names are retained during model training.

- **Labeling:** Replacing the original labels (`Pesar` and `Dokhtar`) with a standardized `Gender` label to simplify the classification task.

- **Cleaning:** Removing duplicates and handling noisy data. For instance, names that appeared more than once with different genders were carefully reviewed. The data was then adjusted to reflect the most accurate gender classification.

- **Augmentation:** Given the relatively small size of the dataset, data augmentation was employed to improve the model's robustness. Several augmentation techniques were considered:
    - **Character Substitution:** Swapping out similar-sounding characters (e.g., "س" with "ث", or "ک" with "گ") to generate new but plausible names.
    - **Minor String Modifications:** This approach, which involved adding or removing characters to create variations of existing names, was not used. It was deemed too risky, as minor changes in Persian names can result in significant gender shifts (e.g., "ایمان" (Iman) to "ایمانه" (Imaneh)).
    - **Augmentation Focus:** Augmentation was applied specifically to the training set to avoid contaminating the test data with artificial noise.

### Before and After Cleaning

| **Before Cleaning** | **After Cleaning** |
|---------------------|--------------------|
| ![image](https://github.com/user-attachments/assets/0bd56e3d-dabd-4da5-af87-540985f40a65) | ![image](https://github.com/user-attachments/assets/60def4dc-7123-4b67-86f2-cbb6c85469b8) |


## Exploratory Data Analysis (EDA)

### Dataset Distribution

| ![image](https://github.com/user-attachments/assets/c1acbccc-d6e3-4811-b125-9bc57f460d73) | ![image](https://github.com/user-attachments/assets/ae12c721-58b9-45b2-a219-41b2627bb04b) |
|-----------------------|----------|
| ![image](https://github.com/user-attachments/assets/1a369046-e018-4812-89ca-9f8425d15b33) | ![image](https://github.com/user-attachments/assets/adcd16bc-7837-4b0e-9464-c5acf993ce25) |

### Data Challenges
The dataset is small compared to similar tasks in other languages, and the primary challenge is dealing with rare names that are difficult to classify. Additionally, the imbalanced distribution of genders, particularly with fewer neutral names, presents challenges for model training and evaluation.

## Model Development

We explored and developed several models:

1. **Character-level Bi-LSTM:**
   - **Binary Classification:** Focused on predicting whether a name is male or female.
   - **Multi-Class Classification:** Included a third class for neutral names, which are neither distinctly male nor female.
   - **Augmentation:** Data augmentation was used to enhance the diversity of the training set, making the model more resilient to variations in name spellings.
   - **Best Practices:** The model employed cross-entropy loss for classification, the Adam optimizer for efficient learning, and learning rate decay to gradually reduce the learning rate during training.

2. **Hazm-Based Models:**
   - Utilized Persian word embeddings from the Hazm library, which captures the semantic meaning of Persian names by representing them as vectors in a high-dimensional space.
   - **Dimensionality Reduction:** PCA was applied to reduce the embeddings' dimensionality from 300 to 50, balancing computational efficiency and model performance.
   - **Binary and Multi-Class Classification:** The Hazm-based models were evaluated for both binary and multi-class classification tasks. 
   - **Results:** The Hazm-based models performed well overall, but they struggled with rare names and neutral gender predictions, indicating a potential limitation in the embedding space or the reduced dimensionality.

3. **FaBERT-Based Models:**
   - Leveraged FaBERT, a BERT-based model fine-tuned specifically for the Persian language, to extract contextual embeddings for each name. FaBERT's deep understanding of Persian context provided more nuanced embeddings.
   - **Dimensionality Reduction:** As with the Hazm models, PCA was applied to reduce the embeddings' dimensionality from 768 to 50.
   - **Binary and Multi-Class Classification:** FaBERT-based models were tested on both binary and multi-class classification tasks.
   - **Results:** FaBERT-based models outperformed Hazm-based models, particularly in predicting neutral names, due to FaBERT's superior ability to capture the linguistic nuances of Persian.

## Results

### Model Performance
- **Binary Classification:**
  - The binary classification models (across Bi-LSTM, Hazm-based, and FaBERT-based) all achieved high accuracy in distinguishing between male and female names. However, certain rare names posed challenges, reducing model confidence in some cases.

- **Multi-Class Classification:**
  - Multi-class classification, which included the neutral gender, proved more challenging. The models tended to be biased towards predicting male or female over neutral, leading to lower accuracy for the neutral class.

### Hazm-Based vs. FaBERT-Based Models
- **Hazm-Based Models:**
  - **Strengths:** Good general performance, especially with names that had clear gender distinctions. The reduced dimensionality via PCA made the models more computationally efficient.
  - **Weaknesses:** Struggled with predicting neutral gender and names with less common usage patterns, likely due to the limitations in the embedding space and reduced feature set.

- **FaBERT-Based Models:**
  - **Strengths:** Demonstrated superior performance, particularly in handling neutral names, due to FaBERT's richer contextual embeddings that capture subtle nuances in the Persian language.
  - **Weaknesses:** Computationally more expensive compared to Hazm-based models, due to the higher dimensionality of the original embeddings. However, the accuracy gains justified this trade-off.

| **Model**          | **Result**                          |
|-------------------------|-------------------------------------|
| **Bi-LSTM (Binary)**    | ![image](https://github.com/user-attachments/assets/746a5985-de02-4419-b602-27850e712899) ![image](https://github.com/user-attachments/assets/55e17f7c-0593-48cd-bdf7-7041ec742d62) |
| **Bi-LSTM (Multi-Class)** | ![image](https://github.com/user-attachments/assets/0014a365-4b86-4e35-b9ff-d382baab3f21) ![image](https://github.com/user-attachments/assets/bced0eb6-12c2-4bba-b09e-74dc08ce3c54) |
| **Hazm-Based Models (Binary)** | ![image](https://github.com/user-attachments/assets/2fc03634-d6ce-40b9-a768-48df0b79485f) |
| **Hazm-Based Models (Multi-Class)** | ![image](https://github.com/user-attachments/assets/a62448b2-bbe0-4664-be79-97bd1e1a3783) |
| **FaBERT-Based Models (Binary)** | ![image](https://github.com/user-attachments/assets/6e00ca90-7bd0-4a0e-8001-fc62bb0420b7) |
| **FaBERT-Based Models (Multi-Class)** | ![image](https://github.com/user-attachments/assets/2537ae75-903d-43cd-b24e-4fdd3fb1a37f) |

### Critical Model Analysis
Given that this model could be deployed publicly, methods such as Focal Loss and Weighted Loss were explored to handle the misclassification of neutral names, thereby increasing the model's reliability. Focal Loss in particular was effective in reducing the bias towards the majority classes (male and female), improving the model's ability to correctly identify neutral names.

## Future Work

- **Regularization:** Implement additional regularization techniques, such as dropout and batch normalization, to further reduce overfitting, particularly in the FaBERT-based models.
- **Threshold Tuning:** Fine-tune classification thresholds to improve the accuracy of neutral name predictions, particularly in edge cases where the model's confidence is low.
- **Model Refinement:** Experiment with advanced architectures and hyperparameter tuning, such as exploring different layers and attention mechanisms in Bi-LSTM models, to push model performance even further.

## References

- [Predicting Gender by First Name Using Character-level Machine Learning](https://arxiv.org/abs/2106.10156)
- [Predicting the Gender of Indonesian Names](https://arxiv.org/abs/1707.07129)
- [What's in a Name? -- Gender Classification of Names with Character Based Machine Learning Models](https://arxiv.org/abs/2102.03692)
- [Character-Level LSTMs for Gender Classification from Name](https://maelfabien.github.io/machinelearning/NLP_7)
- [Hazm - Persian NLP Toolkit](https://github.com/roshan-research/hazm)
- [FaBERT: Pre-training BERT on Persian Blogs](https://arxiv.org/abs/2402.06617)

## Contact

If you have any questions or feedback, feel free to contact [My Email](mailto:shayankebriti@gmail.com).
