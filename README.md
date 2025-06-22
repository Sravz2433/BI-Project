# 🧬 Rett Syndrome Pathogenicity Classifier

This repository implements a hierarchical machine and deep learning framework for the classification of **MECP2 gene mutations** as *pathogenic* or *benign*. MECP2 variants are implicated in **Rett Syndrome** and other neurodevelopmental disorders. The proposed workflow integrates domain-informed features, statistical insights, and multiple learning paradigms to improve classification performance.

---

## 🧪 Project Overview

* **Objective**: To predict the pathogenicity of MECP2 gene variants using a biologically informed feature set and various ML/DL models.
* **Dataset**: Curated from **ClinVar**, containing annotated MECP2 variants with clinical significance labels.
* **Problem Type**: Supervised Binary Classification
* **Techniques Applied**:
  * Classical Machine Learning Models (Logistic Regression, KNN, SVM, etc.)
  * Deep Learning Models (ANN, LSTM, Autoencoder)
  * Data Imbalance Handling using **SMOTE**
  * Hyperparameter Tuning via **Bayesian Optimization with Optuna**

---

## 📂 Dataset Description

* **Source**: [ClinVar Database](https://www.ncbi.nlm.nih.gov/clinvar/)
* **Total Samples**: 1215
  * Pathogenic / Likely Pathogenic: 583
  * Benign / Likely Benign: 632
* **Included Mutation Types**: Deletion, Duplication, SNV, Insertion
* **Format**: Cleaned and structured `.csv` after preprocessing and annotation

---

## 🔄 Methodology

### 📥 Data Preprocessing

* Removed *Variants of Uncertain Significance (VUS)*
* Label Mapping:
  * `0` → Benign/Likely Benign
  * `1` → Pathogenic/Likely Pathogenic
* One-hot and label encoding for categorical variables
* Normalized continuous variables using **MinMaxScaler**

### 🧬 Feature Extraction

Biologically driven features were extracted and engineered to enhance signal:

| Category           | Features Extracted                                       |
| ------------------ | -------------------------------------------------------- |
| **Molecular**      | Mutation consequence, sequence context                   |
| **Genomic**        | Position (GRCh38), exon region, splice site proximity    |
| **Sequence-based** | GC content, flanking bases, mutation type, trinucleotide |
| **Annotation**     | GTF-mapped region labels (e.g., exon, intron)            |
| **Derived**        | Position binning, one-hot bases before/after mutation    |

---

## 🤖 Machine Learning Model Results

| Model                  | Accuracy | Precision | Recall | F1 Score |
| ---------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression    | 88%      | 89%       | 88%    | 88%      |
| K-Nearest Neighbors    | 94%      | 94%       | 94%    | 94%      |
| Support Vector Machine | 92%      | 92%       | 92%    | 92%      |
| Decision Tree          | 95%      | 95%       | 95%    | 95%      |
| Random Forest          | 95%      | 95%       | 95%    | 95%      |
| XGBoost                | 96%      | 96%       | 96%    | 96%      |
| CatBoost               | 96%      | 96%       | 96%    | 96%      |
| Gradient Boosting      | 97%      | 97%       | 97%    | 97%      |

---

## 🧠 Deep Learning Model Results

| Model       | Accuracy | Precision | Recall | F1 Score |
| ----------- | -------- | --------- | ------ | -------- |
| ANN         | 94.65%   | 92.62%    | 96.58% | 94.56%   |
| LSTM        | 94.23%   | 92.56%    | 95.73% | 94.12%   |
| Autoencoder | 93.83%   | 93.86%    | 93.83% | 93.83%   |

---

## 📊 Tools & Libraries

* **ML**: Scikit-learn, XGBoost, LightGBM, CatBoost
* **DL**: PyTorch
* **Optimization**: Optuna (Bayesian Optimization)
* **Data Processing**: Pandas, NumPy
* **Balancing**: SMOTE
* **Visualization**: Matplotlib, Seaborn
* **Genomics**: Biopython, GTF parsing via Pandas

---

## 🧠 Key Contributions

* Domain-informed feature engineering (splice site distance, region labels, GC content)
* SHAP analysis for model interpretability
* Performance evaluation across a wide range of ML and DL models
* Publicly reproducible code pipeline with modular scripts and notebooks

---

## 🧬 Clinical Relevance

* Assists geneticists and clinicians in early detection of disease-linked MECP2 variants
* Supports prioritization of pathogenic mutations in clinical exome pipelines
* A step toward scalable variant classification in **precision medicine**

---

## 📁 Repository Structure

```
📁 MECP2-Analysis/
│
├── data/                        # Sample or cleaned variant datasets
├── Data_Prep_&_Preprocessing/
│   ├── Data_Preperation.ipynb
│   └── Preprocessing.ipynb
├── Feature_Extraction/
│   └── Feature_Extraction.ipynb
├── models/
│   ├── dl_models/
│   │   ├── Encoder.ipynb
│   │   ├── LSTM.ipynb
│   │   └── Neural_Network.ipynb
│   └── ml_models/
│       ├── catboost_info/
│       ├── pkl_files/
│       ├── results/
│       └── ml_models.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── venv/
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Sravz2433/MECP2-Analysis.git
cd MECP2-Analysis

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📜 License

Licensed under the **GNU General Public License v3.0**.
🔗 [View License](LICENSE)

---

## 📫 Contact

Feel free to reach out with questions or collaborations:

* 📧 [sravyasri2433@gmail.com](mailto:sravyasri2433@gmail.com)
* 💼 [LinkedIn – Sravya Sri Mallampalli](https://www.linkedin.com/in/sravya-sri-mallampalli/)

