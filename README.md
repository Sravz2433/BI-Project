# 🧬 Rett Syndrome Pathogenicity Classifier
This repository contains the implementation of a machine learning framework that classifies MECP2 gene mutations as benign or pathogenic. MECP2 mutations are associated with Rett Syndrome and other neurodevelopmental disorders. The proposed pipeline integrates genomic features, statistical analysis, and machine learning, deep learning models to enhance mutation classification accuracy.

---

## 🧪 Project Overview

- **Objective**: To classify whether a given MECP2 mutation is benign or pathogenic using various ML and DL techniques.
- **Dataset**: Extracted from publicly available genomic repositories. It includes labeled MECP2 mutation samples with relevant biological features.
- **Problem Type**: Binary Classification.
- **Techniques Used**: 
  - Machine Learning Models (Logistic Regression, KNN, SVM, etc.)
  - Deep Learning Models (ANN, LSTM, Autoencoder)
  - Data balancing using Oversampling
  - Hyperparameter tuning using **Bayesian Optimization with Optuna**

---

## 🧪 Dataset
* Source: ClinVar Database

* Total variants: 1215

   - Pathogenic/Likely Pathogenic: 583

   - Benign/Likely Benign: 632

* Filter: Deletion Duplication SNV Insertion

* Format: Tabular, converted to .csv using custom scripts.

---

## 🔧 Methodology

### 📥 Data Collection & Cleaning
- Filtered **germline SNVs (Single Nucleotide Variants)** relevant to MECP2.
- Removed **Variants of Uncertain Significance (VUS)** to ensure only clear benign/pathogenic labels.
- Converted raw genomic mutation tables into structured `.csv` format for downstream processing.

### 🧹 Preprocessing
- Applied **One-Hot Encoding** to categorical features.
- Used **Min-Max Normalization** to scale numerical features between 0 and 1.
- Implemented **Label Encoding**:
  - `0` → Benign  
  - `1` → Pathogenic

### 🧬 Feature Extraction
Extracted biologically relevant features using a domain-driven approach:
- **Mutation Type**: e.g., missense, nonsense, silent
- **Transition/Transversion Status**: nucleotide substitution class
- **Flanking Nucleotide Sequences**: bases adjacent to mutation
- **Trinucleotide Context**: three-base window centered on mutation
- **Mutation Position**: genomic location (GRCh38 / GRCh37 coordinates)

### 🛠️ Feature Engineering

| Feature Category     | Features Included                                   |
|----------------------|-----------------------------------------------------|
| **Molecular**         | Mutation type, Consequence                         |
| **Genomic Context**   | Flanking sequences, Trinucleotide context          |
| **Statistical**       | Transition/Transversion classification             |
| **Position-based**    | Genomic coordinates (GRCh37 / GRCh38 references)   |

> These features were engineered to capture both the molecular impact and the contextual positioning of each variant, ensuring a comprehensive understanding of MECP2 mutation behavior.


---

## 🤖 Machine Learning Models Performance

| Model             | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.88     | 0.89      | 0.88   | 0.88     |
| KNN               | 0.94     | 0.94      | 0.94   | 0.94     |
| SVM               | 0.92     | 0.92      | 0.92   | 0.92     |
| Decision Tree     | 0.95     | 0.95      | 0.95   | 0.95     |
| Random Forest     | 0.95     | 0.95      | 0.95   | 0.95     |
| XGBoost           | 0.96     | 0.96      | 0.96   | 0.96     |
| AdaBoost          | 0.94     | 0.94      | 0.94   | 0.94     |
| CatBoost          | 0.96     | 0.96      | 0.96   | 0.96     |
| Gradient Boosting | 0.97     | 0.97      | 0.97   | 0.97     |

---

## 🧠 Deep Learning Models Performance

| Model     | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| ANN       | 0.9465   | 0.9262    | 0.9658 | 0.9456   |
| LSTM      | 0.9423   | 0.9256    | 0.9573 | 0.9412   |
| Autoencoder | 0.9383 | 0.9386    | 0.9383 | 0.9383   |

---

## 🧪 Tools & Libraries Used

- Python
- **Machine Learning**: Scikit-learn, XGBoost, CatBoost, LightGBM
- **Deep Learning**: PyTorch
- **Data Processing**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Optimization**: Optuna (Bayesian Optimization)
- **Data Balancing**: SMOTE

---

## 🧠 Key Learnings

- Effective classification of genomic data using ensemble ML and DL models.
- Oversampling plays a crucial role in imbalanced bioinformatics datasets.
- Bayesian hyperparameter tuning provides more optimal configurations than traditional grid/random search.

---

## 🧬 Domain Impact

By automating the classification of MECP2 mutations:
- We aid clinical researchers in narrowing down pathogenic variants.
- Provide a scalable bioinformatics framework for gene-level mutation analysis.
- Set the stage for future diagnostics in personalized medicine.

---

## 📁 Repository Structure
```
📁 Bioinformatics-MECP2-Classifier/
│
├── 📁 data/
│
├── 📁 Data_Prep_&_Preprocessing/
│   ├── 📄 Data_Preperation.ipynb
│   └── 📄 Preprocessing.ipynb
│
├── 📁 Feature_Extraction/
│   └── 📄 Feature_Extraction.ipynb
│
├── 📁 models/
│   ├── 📁 dl_models/
│   │   ├── 📄 Encoder.ipynb
│   │   ├── 📄 LSTM.ipynb
│   │   └── 📄 Neural_Network.ipynb
│   │
│   └── 📁 ml_models/
│       ├── 📁 catboost_info/
│       ├── 📁 pkl_files/
│       ├── 📁 results/
│       └── 📄 ml_models.ipynb
│
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 LICENSE
└── 📁 venv/
```
---

## 📜 License

This project is licensed under the **GNU General Public License v3.0**.  
🔗 [View License](LICENSE)

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Sravz2433/BI-Project.git
cd mecp2-mutation-classification

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # For Unix
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📫 Contact

For queries, feel free to reach out at:
- 📧 sravyasri2433@gmail.com
- 🔗 LinkedIn: [Sravya Sri Mallampalli](https://www.linkedin.com/in/sravya-sri-mallampalli/)

